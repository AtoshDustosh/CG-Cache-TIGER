import torch
from torch import Tensor

import numpy as np
from collections import OrderedDict, deque
from typing import List, Tuple

from .graph import Graph

from torch.profiler import record_function


class CGCache:

    slots_nids: List[Tensor]
    slots_eids: List[Tensor]
    slots_tss: List[Tensor]
    offsets: Tensor

    def __init__(
        self,
        cap: int,
        n_neighbor: int,
        n_layer: int,
        graph: Graph,
        evict_policy: str = "lru",
        device: torch.device = torch.device("cpu"),
    ):
        # OPT All large tensors should be pre-allocated here and reused
        self.cap = cap
        self.policy = evict_policy
        self.device = device

        self.L = n_layer
        self.k = n_neighbor
        self.graph = graph

        # Rotating cache slots for all layers
        self.slots_nids = [torch.zeros([])]
        self.slots_tss = [torch.zeros([])]
        self.slots_eids = [torch.zeros([])]
        self.offsets = torch.zeros([self.cap], device=self.device, dtype=torch.long)

        # Initialize cached layers: layer 1 -> layer L
        for l in range(1, self.L + 1):
            self.slots_nids.append(
                torch.zeros([self.cap, self.k**l], device=self.device, dtype=torch.long)
            )
            self.slots_eids.append(
                torch.zeros([self.cap, self.k**l], device=self.device, dtype=torch.long)
            )
            self.slots_tss.append(
                torch.zeros(
                    [self.cap, self.k**l], device=self.device, dtype=torch.float
                )
            )

        # Cache slot index
        # We use a CPU dict because querying for a single batch is usually fast
        self.mapping = OrderedDict()  # map: nid -> cache slot
        self.free_slots = deque(range(self.cap))

        print(f"cg cache initialized with cap {self.cap}")
        pass

    def debug_msg(self) -> str:
        msg = f"cap: {self.cap}\n"
        msg += f"n_cached: {len(self.mapping)}\n"
        msg += f"n_free_slots: {len(self.free_slots)}\n"
        return msg

    @property
    def capacity(self) -> int:
        return self.cap

    @property
    def n_cached(self) -> int:
        ret = len(self.mapping)
        assert ret <= self.cap, "Unexpected cache behavior: exceeding capacity!"
        return ret

    def neg_sample(self, size):
        return np.random.choice(
            np.array(list(self.mapping.keys()), dtype=int), size=size, replace=True
        )

    def reset(self):
        """Clear cache."""
        self.mapping.clear()
        self.free_slots = deque(range(self.cap))
        pass

    @torch.no_grad()
    def get(self, nids: np.ndarray):
        """
        Given some node ids, find their cached CGs cand organize them into proper format.

        Args:
            nids: node ids to be queried
        Returns:
            {List[List[Tensor]]} Layers of CGs [neigh_nids, neigh_eids, neigh_tss]. Located on the device of the cache module.
            {np.ndarray} Mask for cached nodes.
            {int} The number of cached nodes.
        """
        cached_layers: List = [[] for _ in range(self.L + 1)]
        cached_mask = np.zeros_like(nids) < 0

        # Find cached nids
        for i, nid in enumerate(nids):
            if nid in self.mapping:
                # IDEA We should not update cache states here 
                # self.mapping.move_to_end(nid)
                cached_mask[i] = True
            pass
        cached_nids = nids[cached_mask]
        n_cached = len(cached_nids)

        # Map cached nids to cached slots
        cached_slots = np.array([self.mapping[_] for _ in cached_nids])
        cached_slots = torch.from_numpy(cached_slots).long().to(self.device)

        # Initialize the 0-th layer
        cached_nids = torch.from_numpy(cached_nids).long().to(self.device)
        cached_layers[0] = [cached_nids, [], []]
        # Gather other L layers of CGs
        for l in range(1, self.L + 1):
            # This should be much faster than enumerating cached nids
            index = (
                torch.arange(start=0, end=self.k**l, device=self.device).unsqueeze(0)
                + (self.offsets[cached_slots] * self.k ** (l - 1)).unsqueeze(1)
            ) % (self.k**l)

            # "L + 1 - l" is used to match TIGER's twisted cg design
            cached_layers[self.L + 1 - l] = [
                self.slots_nids[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_eids[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_tss[l][cached_slots].gather(1, index).view(-1, self.k),
            ]
            pass

        """
        # [Obsolete] While this looks like the original sampling process, the number of cached_nids is smaller than that of raw neighbors by 1 order of magnitude
        for l in range(self.L):
            step = self.k**l
            layer_nids = np.zeros([bs_cached * step], dtype=int)
            layer_eids = np.zeros([bs_cached * step], dtype=int)
            layer_tss = np.zeros([bs_cached * step], dtype=int)
            for i, nid in enumerate(cached_nids):
                base = i * step
                slot_idx = self.mapping[nid]
                startpos = self.offsets[slot_idx]
                if startpos == 0:
                    layer_nids[base : base + step] = self.slots_nids[l][:]
                    layer_eids[base : base + step] = self.slots_eids[l][:]
                    layer_tss[base : base + step] = self.slots_tss[l][:]
                else:
                    layer_nids[base : base + step - startpos] = self.slots_nids[l][
                        startpos:
                    ]
                    layer_nids[base + step - startpos : base + step] = self.slots_nids[
                        l
                    ][:startpos]
                    layer_eids[base : base + step - startpos] = self.slots_eids[l][
                        startpos:
                    ]
                    layer_eids[base + step - startpos : base + step] = self.slots_eids[
                        l
                    ][:startpos]
                    layer_tss[base : base + step - startpos] = self.slots_tss[l][
                        startpos:
                    ]
                    layer_tss[base + step - startpos : base + step] = self.slots_tss[l][
                        :startpos
                    ]
            cached_layers.append([layer_nids, layer_eids, layer_tss])
        """

        return cached_layers, cached_mask, n_cached

    @torch.no_grad()
    def put_uncached(self, uncached_layers: List[List[Tensor]]):
        """
        Store the uncached layers.
        If there is no available cache slots, replace with built-in evict policy.

        Args:
            layers: Uncached layers of CGs.
        """
        uncached_nids = uncached_layers[0][0].cpu().numpy()

        # FIX This is not LRU because you cut out some nodes.
        # When len(uncached_nids) > cap, we only cache a part of those nodes
        n_uncached = min(len(uncached_nids), self.cap)

        # Find replacable cache slots
        uncached_slot_ids = np.zeros([n_uncached], dtype=int)
        n_free = 0
        n_evicted = 0
        while self.free_slots and n_free < n_uncached:
            uncached_slot_ids[n_free] = self.free_slots.pop()
            n_free += 1
        while self.mapping and n_free + n_evicted < n_uncached:
            # OPT Use learnable evict seq generator to evict cache slots in parallel
            uncached_slot_ids[n_free + n_evicted] = self.mapping.popitem(last=False)[1]
            n_evicted += 1
        assert (
            n_free + n_evicted == n_uncached
        ), "Unexpected cache behavior: incomplete put"

        # Update mapping
        for _, (nid, slot) in enumerate(zip(uncached_nids, uncached_slot_ids)):
            if nid in self.mapping:
                # EXPLAIN Duplicated nid->slot are overwritten, leaving previous cache slots lost. We need to retrieve them in the free slot queue.
                self.free_slots.append(self.mapping[nid])
            self.mapping[nid] = slot
            self.mapping.move_to_end(nid)
            pass
        
        # Cache L layers of CGs in their slots
        uncached_slot_ids = torch.from_numpy(uncached_slot_ids).to(self.device)

        # Reset offsets for replaced slots
        self.offsets.scatter_(0, uncached_slot_ids, torch.zeros_like(uncached_slot_ids))

        # Scatter uncached layers to selected slots
        for l in range(1, self.L + 1):
            # Assign uncached layers
            index = uncached_slot_ids.unsqueeze(-1).expand(-1, self.k**l)

            self.slots_nids[l].scatter_(
                0,
                index,
                # "L + 1 - l" is used to match TIGER's twisted cg design
                # ".view([-1, k**l])" is, again, because of TIGER's design for CGs.
                # layer 0: [600], layer 1: [600, 10], layer 2: [6000, 10], ...
                # TIGER arranges timestamps from left to right (old to new)
                uncached_layers[self.L + 1 - l][0].view([-1, self.k**l])[
                    -n_uncached
                    * self.k**l :
                    # OPT We need a strategy for choosing which nodes to cache instead of the brute-force [:n_uncached] truncating
                    # OBSOLETE: :n_uncached
                ],
            )
            self.slots_eids[l].scatter_(
                0,
                index,
                uncached_layers[self.L + 1 - l][1].view([-1, self.k**l])[
                    -n_uncached * self.k**l :
                ],
            )
            self.slots_tss[l].scatter_(
                0,
                index,
                uncached_layers[self.L + 1 - l][2].view([-1, self.k**l])[
                    -n_uncached * self.k**l :
                ],
            )
            pass

        pass

    @torch.no_grad()
    def update_cached(
        self, srcs: np.ndarray, dsts: np.ndarray, tss: np.ndarray, eids: np.ndarray
    ):
        """
        Update only cached CGs with new batch data.
        Uncached nodes are skipped.

        TODO Add an option for directed and undirected graphs.

        Args:
            srcs: source nodes.
            dsts: destination nodes.
            tss: ascendingly sorted timestamps of the events.
            eids: event ids of the events.
        """

        def parallel_update():
            """
            # OPT The following procedure can be implemented in an more efficient way:
            1. Collect updatables with pre-defined 1D tensor with shape [k]. Rotate the index if the number of collected neighbors exceeds k. Use gather to align them to the left and fill the rest with 0s.
            2. Pass the collected updatables to GPU and prepare the 1-st layer with parallel tensor operations according to slot offsets.
            3. Sample L-1 layers for the collected updatables. Then do the same as step 2.
            """

            # TODO Use torch sort+unique_consecutive+cumsum+gather
            # Step 1: collect "updatable" cache slots
            updatables = {}
            for _, (src, dst, eid, ts) in enumerate(zip(srcs, dsts, eids, tss)):
                # src -> dst
                if src in self.mapping:
                    if src in updatables:
                        updatables[src][0].append(dst)
                        updatables[src][1].append(eid)
                        updatables[src][2].append(ts)
                    else:
                        updatables[src] = [[], [], []]
                        updatables[src][0].append(dst)
                        updatables[src][1].append(eid)
                        updatables[src][2].append(ts)
                # dst -> src
                if dst in self.mapping:
                    if dst in updatables:
                        updatables[dst][0].append(src)
                        updatables[dst][1].append(eid)
                        updatables[dst][2].append(ts)
                    else:
                        updatables[dst] = [[], [], []]
                        updatables[dst][0].append(src)
                        updatables[dst][1].append(eid)
                        updatables[dst][2].append(ts)

            # TODO Don't use padding. Padded 0s trigger too many func calls.
            # Step 2: prepare the 0th and 1st layer (truncation + 0-padding)
            n_updated = len(updatables)
            all_updates_nids = np.zeros([n_updated * self.k], dtype=int)
            all_updates_eids = np.zeros([n_updated * self.k], dtype=int)
            all_updates_tss = np.zeros([n_updated * self.k], dtype=float)
            updated_nids = []
            for idx, (updated_nid, updates) in enumerate(updatables.items()):
                len_overwrite = min(len(updates[0]), self.k)
                updates_nids, updates_eids, updates_tss = updates
                # Truncate unnecessary neighbors
                all_updates_nids[idx * self.k : idx * self.k + len_overwrite] = (
                    updates_nids[:len_overwrite]
                )
                all_updates_eids[idx * self.k : idx * self.k + len_overwrite] = (
                    updates_eids[:len_overwrite]
                )
                all_updates_tss[idx * self.k : idx * self.k + len_overwrite] = (
                    updates_tss[:len_overwrite]
                )
                updated_nids.append(updated_nid)
            updated_nids = np.array(updated_nids)  # [n_updated]

            # Initialize the first two layers of incremental CGs
            updated_layers: List[Tuple] = [(updated_nids, np.array([]), np.array([]))]
            updated_layers.append((all_updates_nids, all_updates_eids, all_updates_tss))

            # Step 3: sample and initialize higher layers (layer depth >= 2)
            for l in range(2, self.L + 1):
                layer_nids, layer_eids, layer_tss, *_ = (
                    self.graph.sample_temporal_neighbor(
                        updated_layers[l - 1][0], updated_layers[l - 1][2], self.k
                    )
                )
                updated_layers.append((layer_nids, layer_eids, layer_tss))
                pass

            # Convert to tensors
            for depth in range(len(updated_layers)):
                neigh_nids, neigh_eids, neigh_tss = updated_layers[depth]
                updated_layers[depth] = (
                    torch.from_numpy(neigh_nids).to(self.device).long(),
                    torch.from_numpy(neigh_eids).to(self.device).long(),
                    torch.from_numpy(neigh_tss).to(self.device).float(),
                )
                pass

            # Step 4: find cache slots
            updated_slots = np.zeros_like(updated_nids)
            for i, nid in enumerate(updated_nids):
                self.mapping.move_to_end(nid)
                updated_slots[i] = self.mapping[nid]
                pass
            updated_slots = torch.from_numpy(updated_slots).to(self.device).long()
            updated_offsets = self.offsets[updated_slots]

            # Step 5: parallel update
            for l in range(1, self.L + 1):
                index = (
                    torch.arange(
                        start=0, end=self.k**l, device=self.device
                    ).unsqueeze(0)
                    - (updated_offsets * self.k ** (l - 1)).unsqueeze(1)
                ) % (self.k**l)
                layer_nids, layer_eids, layer_tss = updated_layers[l]
                layer_nids_rolled = layer_nids.view(-1, self.k**l).gather(1, index)
                layer_eids_rolled = layer_eids.view(-1, self.k**l).gather(1, index)
                layer_tss_rolled = layer_tss.view(-1, self.k**l).gather(1, index)

                mask = layer_nids_rolled != 0

                merged_nids = torch.where(
                    mask, layer_nids_rolled, self.slots_nids[l][updated_slots]
                )
                merged_eids = torch.where(
                    mask, layer_eids_rolled, self.slots_eids[l][updated_slots]
                )
                merged_tss = torch.where(
                    mask, layer_tss_rolled, self.slots_tss[l][updated_slots]
                )

                self.slots_nids[l].index_copy_(0, updated_slots, merged_nids)
                self.slots_eids[l].index_copy_(0, updated_slots, merged_eids)
                self.slots_tss[l].index_copy_(0, updated_slots, merged_tss)
                pass

            pass

        def sequential_update():
            """
            This implementation is extremely slow due to scattered CPU/GPU cross processing, fractured data I/O, and massive number of GPU kernel launches.
            """
            # Collect "updatable" cache slots
            updatables = {}
            for _, (src, dst, eid, ts) in enumerate(zip(srcs, dsts, eids, tss)):
                # src -> dst
                if src in self.mapping:
                    if src in updatables:
                        updatables[src][0].append(dst)
                        updatables[src][1].append(eid)
                        updatables[src][2].append(ts)
                    else:
                        updatables[src] = [[], [], []]
                        updatables[src][0].append(dst)
                        updatables[src][1].append(eid)
                        updatables[src][2].append(ts)
                # dst -> src
                if dst in self.mapping:
                    if dst in updatables:
                        updatables[dst][0].append(src)
                        updatables[dst][1].append(eid)
                        updatables[dst][2].append(ts)
                    else:
                        updatables[dst] = [[], [], []]
                        updatables[dst][0].append(src)
                        updatables[dst][1].append(eid)
                        updatables[dst][2].append(ts)

            # Sample L-1 layers for updatable nodes
            # For each layer, update cache slots
            # Newer events with larger timestamps are appended to the positions by offsets
            # OPT Perform all CPU-intensive operations before moving to GPU
            for root, layer_cg in updatables.items():
                # print(f"Updating {root} with len({len(layer_cg[0])})")
                layer_nids, layer_eids, layer_tss = layer_cg
                layer_nids = np.array(layer_nids)
                layer_eids = np.array(layer_eids)
                layer_tss = np.array(layer_tss)
                reset_offset = len(layer_nids) >= self.k
                slot = self.mapping[root]
                offset = int(self.offsets[slot])
                for l in range(1, self.L + 1):
                    if l != 1:
                        layer_nids, layer_eids, layer_tss, *_ = (
                            self.graph.sample_temporal_neighbor(
                                layer_nids, layer_tss, self.k
                            )
                        )
                    t_layer_nids = torch.from_numpy(layer_nids).to(self.device).ravel()
                    t_layer_eids = torch.from_numpy(layer_eids).to(self.device).ravel()
                    t_layer_tss = torch.from_numpy(layer_tss).to(self.device).ravel()
                    if reset_offset:
                        self.offsets[slot] = 0
                        self.slots_nids[l][slot] = t_layer_nids[-self.k**l :]
                        self.slots_eids[l][slot] = t_layer_eids[-self.k**l :]
                        self.slots_tss[l][slot] = t_layer_tss[-self.k**l :]
                    else:
                        """
                        1. Roll the slot by -offset (l-1) units.
                        2. Replace some neighbor units.
                        3. Roll back the slot by offset (l-1) units.
                        """
                        tmp_nids = self.slots_nids[l][slot].roll(
                            -(offset * self.k ** (l - 1))
                        )
                        tmp_nids[: len(t_layer_nids)] = t_layer_nids
                        self.slots_nids[l][slot] = tmp_nids.roll(
                            offset * self.k ** (l - 1)
                        )

                        tmp_eids = self.slots_nids[l][slot].roll(
                            -(offset * self.k ** (l - 1))
                        )
                        tmp_eids[: len(t_layer_eids)] = t_layer_eids
                        self.slots_eids[l][slot] = tmp_eids.roll(
                            offset * self.k ** (l - 1)
                        )

                        tmp_tss = self.slots_nids[l][slot].roll(
                            -(offset * self.k ** (l - 1))
                        )
                        tmp_tss[: len(t_layer_tss)] = t_layer_tss
                        self.slots_tss[l][slot] = tmp_tss.roll(
                            offset * self.k ** (l - 1)
                        )

                        pass
                    pass
                pass

        # sequential_update()
        parallel_update()
        pass
