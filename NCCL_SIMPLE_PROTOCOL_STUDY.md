# NCCL Simple protocol: source study and implications for CRI

## 1. Scope, repository, and reproducibility

- Repository requested: <https://github.com/CaoZhongZ/nccl.git>.
- Branch studied: `master`.
- Exact revision: `d97a32fac84f69cbd022ffdc57d5687c760742c7`.
- Commit subject: `2.18.1-1`; commit date: April 18, 2023.
- Version in `makefiles/version.mk`: NCCL 2.18.1.
- Study date: September 30, 2026.
- Downloaded source: `/tmp/nccl-simple-protocol-source-d97a32f` on the local machine.

This is a source-level study of that snapshot, not of the latest NVIDIA NCCL. All external source links below are pinned to its commit. No CUDA build or NCCL runtime benchmark was performed for this study. The CRI measurements in Section 17 are from the earlier SYCL benchmark runs, not NCCL runs.

The focus is the ordinary Simple FIFO protocol, its GPU implementation, Ring AllReduce, direct-buffer variants, and P2P/SHM/NET transport integration. Tree, CollNet, and NVLS are covered where they change the interpretation of Simple. This is not a complete study of topology discovery, every network plugin, or SHARP hardware internals.

To obtain the same revision independently:

```sh
git clone https://github.com/CaoZhongZ/nccl.git
git -C nccl checkout d97a32fac84f69cbd022ffdc57d5687c760742c7
```

## 2. Executive summary

**NCCL Simple is a bulk-data protocol with out-of-band readiness and credit counters. It is not a payload-plus-flags packet format.**

Its ordinary buffered path has three essential components:

1. A payload FIFO divided into eight logical steps.
2. A producer-published `tail` counter saying which steps are ready.
3. A consumer-published `head` counter returning space for reuse.

GPU workers copy or reduce an entire slice. A local barrier joins those workers with synchronization threads; a posting thread performs a system-scope fence before publishing a nonempty send. Receivers poll the readiness counter and load payload through volatile global-memory instructions. This amortizes synchronization over a much larger amount of data than the LL/LL128 protocols.

The headline engineering consequences are:

- Simple carries no per-line flags inside the ordinary payload FIFO; payload capacity is therefore 100% of that FIFO's data area. This is not 100% physical-link efficiency: counters, packets, memory accesses, and transport overhead still exist.
- The default non-ARM Simple buffer is 4 MiB, yielding 512 KiB per logical step. Ring AllReduce normally reserves two steps per slice and four per chunk: a nominal 1 MiB slice and 2 MiB chunk.
- Actual slices can be much smaller than those nominal capacities. Protocol step advancement remains fixed even for empty slices.
- Separate wait and post thread roles allow copying and fence/publication work to overlap across slices.
- GPU-to-GPU writes, GPU-to-GPU reads, shared host memory, and network transfers can all implement Simple. Simple is not synonymous with PCIe, Ring, or direct writes.
- Direct-buffer paths can bypass FIFO payload copies while retaining pointer exchange and progress coordination.
- CUDA/PTX memory-ordering details are part of the implementation. Copying only its counter logic into a SYCL/Intel kernel would be unsafe.

The current CRI `simple_pcie` implementation is fundamentally different: `Rt64_128_PCIE` embeds readiness flags within its 256-byte messages, leaving 240 bytes of payload. Its name does not make it NCCL Simple.

## 3. Source map and reading order

| Source | What it establishes |
|---|---|
| [Version][version] | Exact release version |
| [AllReduce API][allreduce-api] | Public API feeds chunk/slice constants into enqueueing |
| [Protocol definitions][protocols] | `ProtoSimple` template parameters and bytes per step |
| [Device connection layout][conn-info] | Payload pointers, `head`, `tail`, direct flags, FIFO metadata, saved steps |
| [Host-visible control layout][control-mem] | Separate send/receive control areas and cache-line padding |
| [Simple wait/post logic][simple-wait] | Credit/readiness predicates and progress publication |
| [Simple operation loop][simple-op] | Slice processing, barriers, copy/reduction, empty slices |
| [Connection initialization][simple-connections] | Counter rounding and direct capability selection |
| [Thread roles and teardown][simple-roles] | Wait/post workers and saved state |
| [Direct pointer exchange][simple-direct] | Direct push/pull handshake and reduction arguments |
| [Vectorized reduction/copy][reduce-copy] | Payload load/store paths and alignment handling |
| [PTX memory helpers][memory-helpers] | Volatile, relaxed-system, and fence instructions |
| [Ring AllReduce][ring-allreduce] | Reduce-scatter/all-gather composed from Simple primitives |
| [Tree AllReduce][tree-allreduce] | Split reduction/broadcast execution |
| [Buffer defaults][buffer-defaults] | Default capacity and configuration |
| [Host algorithm selection][algo-selection] | Model-driven algorithm/protocol choice |
| [Host chunk/proxy parameters][host-chunks] | Matching GPU and proxy step schedules |
| [P2P setup][p2p-setup] and [P2P connection][p2p-connect] | Read/write buffer placement and counter mapping |
| [SHM connection][shm-connect] and [SHM progress][shm-progress] | Mapped host buffers and copy-engine staging |
| [NET send progress][net-send] and [NET receive progress][net-recv] | Network request completion, credit return, visibility flush |
| [Send/Recv kernels][sendrecv] | Point-to-point use of `ProtoSimple<1,1>` |
| [Performance model][tuning] | Thread limits, protocol filtering, latency/bandwidth estimates |

Read `primitives.h`, `devcomm.h`, and `prims_simple.h` first. Then read `all_reduce.h` for algorithm composition, and only then the transport files. Reading Ring AllReduce alone hides the flow-control and memory-ordering contract.

## 4. Protocol, algorithm, and transport are different layers

| Layer | Examples | Main responsibility |
|---|---|---|
| Collective | AllReduce, AllGather, ReduceScatter, Broadcast | Mathematical operation and result layout |
| Algorithm | Ring, Tree, CollNet, NVLS | Which peers exchange which data, and in what order |
| Protocol | Simple, LL, LL128 | Data representation, readiness detection, transfer granularity |
| Transport | P2P, SHM, NET | Memory placement/mapping and physical transfer/proxy mechanism |

Consequently:

- Ring/Simple can run over PCIe, NVLink, or a network-backed connection.
- Tree also has a Simple implementation.
- Selecting Simple does not force FIFO staging: direct-buffer variants may avoid staging.
- Selecting P2P read mode changes where the Simple FIFO resides, not the definition of the collective.

The Simple specialization is parameterized as:

```text
Primitives<T, RedOp, Fan, Direct,
           ProtoSimple<SlicePerChunk, StepPerSlice, Unroll,
                       MultimemSrcs, MultimemDsts>, P2p>
```

`Fan` bounds receive/send fan-in and fan-out; `Direct` enables direct capabilities; `P2p` distinguishes point-to-point usage. Multimem parameters support NVLS variants. None of these imply that every operation will use a direct user-buffer address.

## 5. Data and control memory

### 5.1 Ordinary FIFO organization

For a private buffered connection:

```text
Simple payload FIFO
  slot 0 | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 | slot 6 | slot 7

Control state, separate from payload
  head: consumed/credited step position
  tail: produced/ready step position
  step: saved per-connection progress between operations
```

The principal connection fields are:

| Field | Meaning |
|---|---|
| `buffs[NCCL_PROTO_SIMPLE]` | Payload base used by this connection |
| `tail` | Counter polled by receive-wait threads and updated by send-post threads |
| `head` | Counter polled by send-wait threads and updated by receive-post threads |
| `step` | Persistent connection position restored on entry and saved on exit |
| `sizesFifo` | Optional GPU-to-proxy byte counts; not payload flags |
| `offsFifo` | Optional proxy-to-GPU offsets into shared payload storage |
| `ptrExchange` | Rendezvous slot for direct-buffer pointers |
| `redOpArgExchange` | Reduction pre-operation argument exchange for direct pulls |

The control structures pad frequently accessed fields onto separate cache-line regions. This reduces control-field interference; it is not a requirement that payload and control be transmitted as one atomic memory transaction.

The `ncclConnInfo` comments describe the normal push arrangement: receive buffers/tail are local to the receiver and send buffers/tail are remote to the sender. **P2P read mode is an explicit exception for the Simple payload buffer**: the sender's FIFO is local to the sender and remotely read by the receiver.

### 5.2 Capacity arithmetic

Let:

```text
B = Simple buffer capacity in bytes
Q = NCCL_STEPS = 8
E = sizeof(T)
stepBytes = B / Q
stepElements = B / Q / E
slot(step) = step modulo Q
ordinaryPayloadAddress = fifoBase + slot(step) * stepElements
```

Default capacities in this snapshot:

| Configuration | FIFO capacity | Bytes per step |
|---|---:|---:|
| Non-ARM default | 4 MiB | 512 KiB |
| ARM default | 1 MiB | 128 KiB |
| Explicit `NCCL_BUFFSIZE` | Configured byte count | Configured count / 8 |

For BF16 or FP16, a non-ARM default step contains 262,144 elements. Buffer allocation scales with the actual connections/channels and transport configuration; 4 MiB is not a communicator-wide total or the maximum collective message size. Shared NET buffers and NVLS mappings need not follow the private-FIFO allocation picture.

The counters are 64-bit step positions, not eight-valued slot identifiers. Modulo eight is used only to find storage. This lets a reused slot be distinguished by its progress generation.

### 5.3 Two-GPU buffer layout

The following layout shows one channel with independent connections in both directions, using the ordinary buffered P2P write path and the non-ARM default 4 MiB Simple FIFO. Addresses increase downward within each allocation; the send and receive allocations are separate. Allocation alignment padding is omitted.

```text
       GPU 0 memory                          GPU 1 memory

  SEND allocation for 0 → 1             SEND allocation for 1 → 0
  ┌─────────────────────────────┐       ┌─────────────────────────────┐
  │ head₀→₁: credit counter     │       │ head₁→₀: credit counter     │
  │ Other fields + padding      │       │ Other fields + padding      │
  │       4 KiB control region  │       │       4 KiB control region  │
  └─────────────────────────────┘       └─────────────────────────────┘
         Separate allocation                   Separate allocation
  RECEIVE allocation for 1 → 0          RECEIVE allocation for 0 → 1
  ┌─────────────────────────────┐       ┌─────────────────────────────┐
  │ tail₁→₀: readiness counter  │       │ tail₀→₁: readiness counter  │
  │ Other fields + padding      │       │ Other fields + padding      │
  │       4 KiB control region  │       │       4 KiB control region  │
  ├─────────────────────────────┤       ├─────────────────────────────┤
  │ LL protocol buffer          │       │ LL protocol buffer          │
  ├─────────────────────────────┤       ├─────────────────────────────┤
  │ LL128 protocol buffer       │       │ LL128 protocol buffer       │
  ├─────────────────────────────┤       ├─────────────────────────────┤
  │ SIMPLE FIFO — 4 MiB         │       │ SIMPLE FIFO — 4 MiB         │
  │ ┌─────────────────────────┐ │       │ ┌─────────────────────────┐ │
  │ │ Slot 0          512 KiB │ │       │ │ Slot 0          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 1          512 KiB │ │       │ │ Slot 1          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 2          512 KiB │ │       │ │ Slot 2          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 3          512 KiB │ │       │ │ Slot 3          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 4          512 KiB │ │       │ │ Slot 4          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 5          512 KiB │ │       │ │ Slot 5          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 6          512 KiB │ │       │ │ Slot 6          512 KiB │ │
  │ ├─────────────────────────┤ │       │ ├─────────────────────────┤ │
  │ │ Slot 7          512 KiB │ │       │ │ Slot 7          512 KiB │ │
  │ └─────────────────────────┘ │       │ └─────────────────────────┘ │
  └─────────────────────────────┘       └─────────────────────────────┘
```

For **GPU 0 → GPU 1**, the payload FIFO and its `tail` live in GPU 1's receive allocation; its `head` lives in GPU 0's send allocation. The readiness and credit counters therefore do not share one control page. Both counters are 64-bit values; the 4 KiB regions contain additional fields and padding. The [control structures][control-mem] define those regions, and [P2P setup/connection code][p2p-setup] lays out the protocol buffers in LL, LL128, Simple order.

GPU 0 polls its local `head`, writes the remote FIFO, and publishes the remote `tail` after payload completion and the system-scope fence. GPU 1 polls its local `tail`, consumes its local FIFO, then writes GPU 0's `head` to return credits. The reverse direction has its own independent counters and FIFO.

For full nominal Ring/Simple transfers starting at slot 0, the FIFO subdivisions are:

```text
Slots:   [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ]
         └─1 MiB─┘ └─1 MiB─┘ └─1 MiB─┘ └─1 MiB─┘
           slice     slice     slice     slice
         └─────2 MiB───────┘ └─────2 MiB───────┘
                chunk              chunk

After slot 7, storage wraps back to slot 0.
```

These are logical subdivisions of one shared FIFO, processed collectively by the channel's worker threads. Simple payload slots contain no embedded readiness flags, and smaller transfers can use less than the nominal slice/chunk capacity. In P2P read mode, the Simple FIFO instead follows the sender's control region and the receiver reads it remotely; the head/tail ownership remains the same.

### 5.4 Channels have separate connection buffers

Section 5.3 shows only one channel. For the ordinary private buffered P2P path, each channel has its own peer connections and their control/FIFO state. The 4 MiB Simple FIFO is per receive connection within a channel, rather than one FIFO shared by every channel on a GPU.

The host-side pointer hierarchy makes the channel dimension explicit:

```text
comm->channels[channel].peers[peer]->recv[connectionIndex]
    .conn.buffs[NCCL_PROTO_SIMPLE]
```

The corresponding connection also holds `head`, `tail`, and saved `step`. The [channel/peer definitions][channel-peers] expose separate send/receive connection records for each peer. A channel is the communication state; for Ring/Simple, one CUDA block normally executes a channel's work.

For two GPUs and two channels, the logical ownership is:

```text
GPU 0 memory                              GPU 1 memory

Channel 0 — executed by block 0           Channel 0 — executed by block 0
  Send connection 0 → 1                    Send connection 1 → 0
    head[0, 0→1]                             head[0, 1→0]
  Receive connection 1 → 0                 Receive connection 0 → 1
    tail[0, 1→0]                             tail[0, 0→1]
    Simple FIFO[0, 1→0]: 4 MiB               Simple FIFO[0, 0→1]: 4 MiB
      [slot 0 ... slot 7]                      [slot 0 ... slot 7]

Channel 1 — executed by block 1           Channel 1 — executed by block 1
  Send connection 0 → 1                    Send connection 1 → 0
    head[1, 0→1]                             head[1, 1→0]
  Receive connection 1 → 0                 Receive connection 0 → 1
    tail[1, 1→0]                             tail[1, 0→1]
    Simple FIFO[1, 1→0]: 4 MiB               Simple FIFO[1, 0→1]: 4 MiB
      [slot 0 ... slot 7]                      [slot 0 ... slot 7]
```

This is a logical channel view; the allocation layout for each connection is the one in Section 5.3. Other control fields, LL/LL128 buffers, and alignment padding are omitted here.

GPU 0's Channel 0 sends into GPU 1's Channel 0 receive FIFO and publishes its tail; Channel 1 uses the Channel 1 FIFO and counters independently. The reverse directions follow the same channel mapping. Thus this example has **8 MiB of Simple receive FIFO storage per GPU**, plus control and other protocol storage.

Each block joins its own workers before posting its own connection's progress. A block barrier does not join another block, and progress on one channel does not publish data or return credit for another. In the proposed SYCL transport, map workgroup `c` to channel `c` while keeping send-control, receive-control, and data in the separate ranges specified in `SYCL_BULK_FIFO_PLAN.md`.

With eight Ring channels and one private incoming connection per channel, the non-ARM default therefore provides **8 × 4 MiB = 32 MiB of Simple receive FIFO storage per GPU**. This is global device-memory storage, reused across operations. Control regions and other protocol buffers add their own allocations.

`NCCL_BUFFSIZE` can change the FIFO capacity. With a configured 2 MiB FIFO, the same eight channels use 16 MiB of Simple payload storage; each of eight steps holds 256 KiB, a nominal Ring slice holds 512 KiB, and a nominal chunk holds 1 MiB. Large messages then require more slices, and therefore more barriers/fences and counter publications. Smaller messages that already fit the slice target need not gain extra publications.

## 6. Steps, slices, and chunks

These terms are easy to conflate:

- **Step:** unit of FIFO credit accounting and slot addressing.
- **Slice:** unit that workers process before a barrier and counter publication. It reserves `StepPerSlice` steps.
- **Chunk:** one primitive operation's data subdivision, processed through `SlicePerChunk` slices.
- **Ring algorithm phase/hop:** collective-level movement of a chunk between peers, not one FIFO step.

Ring AllReduce instantiates:

```text
ALLREDUCE_SLICESTEPS = 8 / 4 = 2
ALLREDUCE_CHUNKSTEPS = 8 / 2 = 4
ProtoSimple<SlicePerChunk=2, StepPerSlice=2>
```

With a default 4 MiB buffer:

```text
nominal step capacity  = 512 KiB
nominal slice capacity = 1 MiB
nominal chunk capacity = 2 MiB
```

The word **nominal** matters. `genericOp` calculates its slice target in elements as:

```text
sliceTarget = max(roundUp(ceil(nelem / SlicePerChunk), 16),
                  stepElements * StepPerSlice / 32)
```

It then clips each nonempty slice to the remaining element count. This expression is equivalent to the source's `divUp(nelem, 16*SlicePerChunk)*16` term. A smaller message or final chunk therefore does not transfer a full nominal slice.

For example, with BF16 and the defaults, a primitive call with 65,536 elements has a 32,768-element target per slice, or 64 KiB. Each slice still advances the protocol by two logical steps. At most four such two-step slices can be outstanding before the eight-step credit window is exhausted.

Broadcast and Reduce use one chunk step and one slice step. Tree AllReduce and point-to-point Simple use `ProtoSimple<1,1>`. The 1 MiB/2 MiB Ring figures must not be generalized to every Simple operation.

### Empty slices are not optional

`genericOp` clamps negative element counts to zero. It nevertheless executes the remaining scheduled slices with zero data, matching barriers and incrementing the connection counters. Empty sends omit the data fence but still publish progress.

Dropping empty slices would desynchronize producer, consumer, and proxy schedules. This is especially relevant when channels/rank partitions extend beyond the tail of the user message.

## 7. Flow control: a producer-consumer credit window

For the ordinary buffered path, define:

```text
s = local starting step for the next slice
k = StepPerSlice
Q = 8
```

The exact wait condition in `waitPeer` is:

```text
while cachedPeerStep + (isSend ? Q : 0) < s + k:
    cachedPeerStep = loadPeerCounter()
    checkAbortPeriodically()
```

Thus:

| Side | Counter polled | Condition allowing progress |
|---|---|---|
| Sender | `head` | `head + 8 >= s + k` |
| Receiver | `tail` | `tail >= s + k` |

The sender is checking available storage; the receiver is checking completed production. A cached counter value avoids a global counter load while already-known progress is sufficient.

After the wait, the designated wait thread publishes the slice's source/destination pointer into group shared memory and advances its private step by `k`. Posting threads independently advance their own private step and publish `head` or `tail` after the full-group barrier. These are different threads' copies of the position: **the code is not advancing one counter twice per slice**.

### 7.1 Concrete wraparound example

Start with `head=0`, `tail=0`, and `k=2`:

| Sender slice start | FIFO start slot | Publication after copy | Required credited head |
|---:|---:|---:|---:|
| 0 | 0 | `tail=2` | At least 0 |
| 2 | 2 | `tail=4` | At least 0 |
| 4 | 4 | `tail=6` | At least 0 |
| 6 | 6 | `tail=8` | At least 0 |
| 8 | 0, reused | `tail=10` | At least 2 |

The fifth slice cannot reuse the first slice's reserved space until the receiver has published `head>=2`. The receiver does not consume the first slice until `tail>=2`.

For ordinary private FIFO communication, this maintains an outstanding reserved-step window no larger than eight. Shared-buffer NET paths deliberately adapt credit publication and offsets; their scheduling must be understood from the proxy code rather than assuming the private-buffer ownership model unchanged.

### 7.2 State across operations

`loadRecvConn` and `loadSendConn` restore `conn->step` and round it up to a multiple of `SlicePerChunk*StepPerSlice`. The receive posting role returns credits for any skipped alignment steps. The primitive destructor saves completed posting positions back to the connection and synchronizes before group state is reused.

Counters therefore persist across collective calls; they are not reset to zero for every call. Device and proxy schedules must agree on both alignment and advancement.

### 7.3 Concrete two-GPU Ring/Simple transfer

Consider one ordinary buffered primitive moving a full 2 MiB chunk from GPU 0 to GPU 1, using the non-ARM default FIFO and starting with `head=tail=0`. Ring/Simple splits the chunk into two nominal 1 MiB slices:

| Slice | Payload bytes | FIFO slots reserved | Tail published after production | Head published after consumption |
|---|---:|---|---:|---:|
| First | 1 MiB | 0–1 | 2 | 2 |
| Second | 1 MiB | 2–3 | 4 | 4 |

For each slice, the worker threads collectively loop over payload packs to copy/reduce the slice. With 512 workers, a full 1 MiB slice gives approximately 2 KiB of payload work per worker. After the full-group barrier, the send-post thread performs the system-scope fence and publishes `tail`. The receiver waits for readiness, processes the slice, then posts `head` after its full-group barrier.

These publications occur once per slice, by two steps. They are not per payload store, worker, warp, or whole 2 MiB chunk. The table gives the values after each production/consumption event; it does not require the sender to wait for credit after every slice. Up to four two-step slices can be outstanding before the eight-step FIFO is full.

For a smaller primitive carrying 128 KiB of BF16 data, the slice calculation produces two 64 KiB slices, but the same counters still advance `0 → 2 → 4`. Each small slice reserves two steps even though it does not fill their capacity. NCCL transfers the actual valid payload; the power-of-two padding rule in the proposed SYCL transport is a separate design choice.

## 8. GPU thread roles and overlap

Simple assigns synchronization duties to selected threads instead of making every payload lane poll a flag.

A worker in this study is one CUDA thread participating in payload copying/reduction. Workers execute in warps, and the worker team processes each slice collectively.

With `ThreadPerSync=8`, the constructor assigns:

| Thread region | Roles |
|---|---|
| First eight-thread group | Receive-wait threads and an input-pointer provider |
| Second eight-thread group | Send-wait threads and an output-pointer provider |
| Penultimate eight-thread group | Receive-post threads |
| Last eight-thread group | Send-post threads |

The peer index is the thread's index within its eight-thread group. Fan-in/fan-out is statically bounded by this role layout. The workers are not a disjoint set excluding all wait threads: wait-role threads also participate in the worker region.

When the primitive can send and at least 64 workers would remain, it reserves a final warp:

```text
nworkers = nthreads - 32
```

Otherwise, all its threads are workers. The source states that the extra warp is intended to overlap the fence with copying. After the all-thread barrier, workers can begin preparation for the next slice while posting threads handle the previous slice's fence/counter update; the next full-group barrier still joins them.

For an ordinary one-receive/one-send primitive with 288 threads:

- Threads 0 and 8 wait for receive readiness and send credit.
- Threads 1 and 9 provide local input/output pointers.
- Threads 272 and 280 post receive/send progress.
- Threads 0–255 participate in the worker region.

`subBarrier()` synchronizes workers after pointer selection and peer waits. `barrier()` joins the complete primitive group after payload work. CUDA named barriers use separate IDs for independent groups; a warp-sized group uses warp synchronization. These group barriers are not a global synchronization across all GPUs or all CTAs.

## 9. Memory ordering and visibility: the critical correctness contract

### 9.1 The ordinary nonempty send sequence

The relevant logical order is:

```text
wait for available send credits and any required receive data
publish source/destination pointers into group shared state
worker sub-barrier
workers copy/reduce payload into destination memory
full-group barrier
send posting thread performs fence_acq_rel_sys()
send posting thread performs st_relaxed_sys_global(tail, nextStep)
```

On CUDA architectures at least SM70, the helpers emit `fence.acq_rel.sys` and `st.relaxed.sys.global.u64`. Older paths use `membar.sys` and a volatile global store. The posting fence is conditional on a send role and `dataStored=true`; zero-data slices do not need a payload-publication fence.

Receive posting occurs after the full-group barrier and publishes the consumed position through a relaxed-system store. That path does **not** call the send-data fence just because it is returning a receive credit. A port must preserve the complete read-before-reuse contract rather than inventing symmetry that is absent from this implementation.

### 9.2 Receiver polling is not a generic C++ acquire load

The ordinary `loadStepValue` path uses `ld.volatile.global.u64`. The source explicitly says volatile is faster than acquire but “not as correct,” and requires reduction/copy payload loads to be volatile to avoid stale L1 data. `common_kernel.h` follows that rule with `ld_volatile_global` payload loads.

This is an important limitation of how to describe the implementation:

- It is not simply a `std::atomic` release/acquire queue.
- C++ `volatile`, PTX `ld.volatile`, system-scope fences, and cache bypass policies are not interchangeable abstractions.
- The source is evidence of the implementation's chosen CUDA/PTX contract, not a language-level proof portable to another architecture.
- Polling a fresh counter but loading stale payload would still violate correctness.

For NVLS on supported SM90/CUDA combinations, `loadStepValue` has a separate `multimem.ld_reduce.acquire.sys.global.min.u64` path selected by `NvlsMinPolling`. Its acquire-min behavior must not be confused with the ordinary volatile poll.

### 9.3 What Simple does not require

The ordinary protocol does not require an entire 1 MiB slice, a warp's stores, or a 256-byte transaction to be atomic. The readiness counter is published only after payload completion and ordering. This is different from using embedded flags to validate data written as smaller messages.

However, “no bulk atomicity requirement” does not eliminate requirements on counter visibility, supported cross-device accesses, ordering scope, transport completion, or progress. Those are the actual dependencies.

## 10. Payload copy and reduction machinery

Simple delegates payload work to `reduceCopy`, which takes source/destination pointer arrays, the number of each, the element count, and reduction pre/post-operation information.

For ordinary non-multimem data, the largest pack size is **16 bytes per thread**, selected only when all participating pointers meet its alignment requirement. The implementation tries the unrolled large-pack path, then large-pack remainder handling, then element-sized paths for remaining or unaligned data.

The underlying ordinary instructions include:

```text
ld.volatile.global.v2.b64
st.global.v2.b64
```

The 16-byte pack is not a 16-byte flag-and-data encoding. It is a payload copy/reduction vector. Likewise, `ProtoSimple::calcBytePerGrain()` returning eight bytes does not define a Simple packet width: its own source comment says that this metric is not queried for Simple.

Workers distribute hunks across warps and lanes, unroll loads/reduction/stores, and can write a computed value to multiple destinations. This supports fused operations such as:

| Primitive | Payload action |
|---|---|
| `send` | Local input → send destination |
| `recv` | Receive source → local output |
| `copySend` | Local input → output and send destination |
| `recvReduceSend` | Receive source + local input → reduced send destination |
| `recvReduceCopy` | Receive source + local input → local output |
| `recvReduceCopySend` | Receive source + local input → local output and send destination |

Pre-operations apply to designated input sources; post-operations apply when the final reduction result is produced. A port must not blindly move pre-scaling or post-division to every forwarding hop. Direct pulls additionally need the peer's reduction pre-operation arguments.

## 11. Ring AllReduce, traced end to end

### 11.1 Host selection and setup

The public AllReduce API supplies the four-step chunk and two-step slice constants. Enqueueing evaluates enabled algorithm/protocol combinations with a modeled time and chooses the minimum eligible result. The model incorporates topology, bandwidth, latency, node/rank counts, and corrections; there is no universal byte threshold that always selects Simple.

Host setup computes:

```text
stepBytes = buffSizes[protocol] / 8
chunkSteps = Ring && Simple ? collectiveChunkSteps : 1
sliceSteps = Ring && Simple ? collectiveSliceSteps : 1
chunkBytes = stepBytes * chunkSteps
```

It also gives proxies a matching number of loops and steps. For LL and LL128, effective chunk sizes are reduced for embedded flags; Simple needs no such payload-capacity correction.

### 11.2 GPU chunk mapping

Each Ring channel gets its predecessor and successor. Nominal loop coverage is:

```text
loopElements = nChannels * nRanks * chunkElements
```

For Simple, final-loop chunks are reduced to fit the remaining message and aligned to a worker-related grain. Within a loop, channel `bid` addresses chunks as:

```text
offset = gridOffset + bid * nRanks * realChunkElements
                     + chunkIndex * realChunkElements
```

This mapping is explicit in `runRing`; it differs from the LL/LL128 mapping in the same function.

### 11.3 Primitive sequence

For `P` ranks, each loop executes:

1. `send`: push an initial local chunk to the next rank.
2. `P-2` calls to `recvReduceSend`: reduce incoming partial sums with local input and forward.
3. `directRecvReduceCopySend`: finish one chunk's reduction, store the final result, and start forwarding that result.
4. `P-2` calls to `directRecvCopySend`: store and forward already-reduced chunks.
5. `directRecv`: place the final incoming result chunk in local output.

The fused third primitive connects reduce-scatter and all-gather. There are `2P-1` primitive calls but `2(P-1)` outgoing chunk transfers per rank. Counting primitive calls as link transfers would be wrong.

### 11.4 Two-rank example

For two ranks with halves `A` and `B`:

```text
Rank 0 sends its B input to rank 1.
Rank 1 sends its A input to rank 0.

Rank 0 combines both A contributions, stores A_sum, sends A_sum to rank 1.
Rank 1 combines both B contributions, stores B_sum, sends B_sum to rank 0.

Rank 0 receives B_sum.
Rank 1 receives A_sum.
```

Each rank sends one half during reduction and one half during gathering, so useful outgoing data totals one full input message per rank. Operations may use direct destinations where available, but the mathematical schedule remains the same.

Reduction result production requests the post-operation; all-gather forwarding does not apply that post-operation again.

## 12. Direct-buffer paths: push, pull, and rendezvous

Simple's direct support changes payload addresses, not just buffer size.

### 12.1 Direct write / push

The receiver provides its user output-buffer pointer through `ptrExchange`; the sender accepts it. A direct send's destination can then be:

```text
remoteOutput + destinationIndex + sliceOffset
```

When the direct receive source is the same as the local destination, the receiver avoids an unnecessary payload copy. If forwarding is required, it can still send from the local output that has just been written by its predecessor.

`waitPeer` skips ordinary FIFO send-credit waiting for enabled direct writes because there is no FIFO payload slot to reserve. The receiver still follows the applicable readiness schedule. This must not be simplified into “direct writes need no synchronization.”

### 12.2 Direct read / pull

The sender provides a source pointer and the receiver pulls from it. Depending on the primitive, the source is local input or an intermediate/output buffer. A direct-read send can be an **empty payload send**: no copy is required because the receiver accesses the source.

The source's explicit wait bypasses are conditional:

```text
skipRecvWait = DirectRecv && hasLocalSource && DirectReadEnabled
skipSendWait = DirectSend && (DirectReadEnabled || DirectWriteEnabled)
```

Do not replace these with an unconditional “direct receive never waits.” Pulling initial input and pulling a produced intermediate result have different dependencies.

### 12.3 Pointer handshake

The provider waits for an empty exchange slot, then publishes an encoded pointer. The acceptor waits for a nonempty slot, decodes or substitutes the registered peer pointer, and clears the slot for reuse. The encoding XORs the pointer with the exchange-slot address so a null user pointer need not collide with the empty-slot representation.

Direct-read setup also exchanges a 64-bit reduction argument as two marked 32-bit halves. These markers distinguish an unpublished argument from a valid zero argument; the receiver reconstructs it and clears the argument slots.

Direct capability depends on connection flags and operation metadata:

- Same-process P2P can advertise direct read/write pointers.
- IPC read/write connections require registered user-buffer metadata for the corresponding direct primitive path in this snapshot.
- Other cases fall back to the communication FIFO.
- Some collective connection-index choices explicitly select direct pulls.

Constructor handshake participation matters even when a thread group later uses only nondirect primitive calls. `runTreeSplit` contains an explicit warning that an apparently unnecessary `Direct=1` is needed so it exchanges pointers with a direct peer constructor; removing it can hang.

### 12.4 Two different meanings of “P2P read”

P2P transport read mode can place the **staging FIFO** in sender memory and have the receiver read it. Direct read can instead pull from a **user or intermediate buffer** supplied by the pointer handshake. These are related optimizations, but are not the same mechanism.

## 13. Transport integration

### 13.1 GPU P2P: writes and reads

For the ordinary write path, `p2pSendConnect` maps the remote receive allocation and assigns:

```text
sender.buff = receiver payload FIFO
sender.tail = receiver tail
sender.head = sender head
receiver.buff = receiver payload FIFO
receiver.tail = receiver tail
receiver.head = sender head
```

Payload/tail traffic is pushed toward the receiver. Credit/head traffic returns toward the sender.

For P2P read mode, Simple's FIFO is appended to the sender allocation instead of the receiver allocation. The sender writes its local FIFO; the receiver maps and reads that FIFO remotely. The same head/tail progress roles coordinate readiness and reuse.

`NCCL_P2P_READ_ENABLE` can override the topology-provided read choice. The source describes a default read preference for Ampere GPUs connected by NVLink; this is not a blanket preference for reads on PCIe or CRI. Legacy IPC, direct pointers, intermediate-rank access, and cuMem mappings are distinct setup variants.

The optional P2P copy-engine path replaces Simple's payload destination with local staging and routes progress through a proxy. It is therefore inaccurate to say that P2P/Simple can never use a CPU proxy or copy engine.

### 13.2 SHM: mapped host buffers or staged copies

SHM connections use mapped shared host-memory payload/control allocations. The send/receive pointers still expose a Simple FIFO and head/tail counters to GPU primitives.

Optional copy-engine paths add device staging:

```text
GPU Simple staging → device-to-host copy → shared host FIFO
shared host FIFO → host-to-device copy → GPU receive staging
```

The proxy records CUDA events and publishes readiness only after the relevant asynchronous copy completes. In these proxy paths, the source explicitly limits copy-engine handling to Simple; other protocols are not processed through the same memcpy loop.

Mapped host memory and GPU P2P memory differ in performance and visibility behavior. The shared protocol interface does not make them equivalent physical paths.

### 13.3 NET send: GPU production to NIC consumption

The ordinary NET send progression is:

1. Prepare a buffer/offset and make its credit available to the GPU.
2. GPU writes the payload, a byte count in `sizesFifo`, and its readiness counter.
3. CPU proxy waits for the byte count and GPU readiness.
4. Proxy submits `isend` for the actual slice byte count.
5. Proxy tests network completion.
6. For a private buffer, proxy advances `head` after completion so the GPU can reuse it.

Returning credits at submission instead of completion would allow the GPU to overwrite a buffer the NIC is still reading. Shared NET buffers use offset assignment and a different credit-management policy; inspect `resources->shared` branches before adapting this sequence.

For Simple, the proxy relies on slice readiness rather than scanning embedded flags. The same source has explicit LL/LL128 flag-scanning branches, making the distinction clear.

### 13.4 NET receive: NIC completion is not always GPU visibility

Receive progression is:

1. Post `irecv` into available receive storage.
2. Test completion and collect received sizes.
3. For nonempty Simple/GDR data when required, perform a visibility flush.
4. Complete the flush request before publishing `tail` to the GPU.
5. Wait for GPU consumption credits before retiring/reusing the receive steps.

The flush can use a GDRCopy-related PCIe read or the network plugin's `iflush`. A CPU memory barrier and optional write-combining store fence are also used when publishing host/GDR-mapped control values.

This is an important portability lesson: **network completion, CPU ordering, and GPU-visible payload completion are separate concerns**. CUDA system fences alone are not a replacement for the required receive-side GPUDirect visibility handling.

## 14. Other Simple algorithm uses

### Tree AllReduce

Tree Simple uses `ProtoSimple<1,1>` rather than Ring's `ProtoSimple<2,2>`. The usual split implementation assigns separate groups to reduction upward and broadcast downward, allowing pipelined overlap. The root combines receive/reduce/store/send work in one primitive group.

Host setup adjusts Tree chunk sizes based on message size, channels, and tree depth. The code has a CUDA-version/architecture-specific fallback to an up/down implementation, so “Tree always uses split execution” is not true for every build of this snapshot.

### CollNet and NVLS

Simple also supplies the data-movement primitives for CollNet and NVLS variants. Their host chunk choices and peer fan-out differ from Ring. NVLS uses multimem source/destination parameters and, where supported, hardware-assisted reduction and minimum-progress polling.

These are not evidence that ordinary Simple performs a hardware reduction or always has acquire-min counter loads. Those features belong to specialized NVLS paths.

### Point-to-point Send/Recv

Send/Recv kernels instantiate `ProtoSimple<1,1>` for the Simple path and use connection index 1. A combined kernel can select LL or Simple for different work groups. Host P2P selection can use LL below a configured threshold when the LL connection buffer exists; otherwise it selects Simple.

Point-to-point chunk sizes are separately configured for network, PCIe, and NVLink paths. Ring AllReduce chunk defaults should not be used to predict Send/Recv transfer granularity.

## 15. Simple versus LL and LL128

| Property | Simple | LL | LL128 |
|---|---|---|---|
| Ordinary payload encoding | Bulk data only | Data and flags in each 16-byte line | 120 data bytes + 8 flag bytes per 128-byte line |
| Payload fraction in that encoding | 100% | 50% | 93.75% |
| Readiness | Separate slice-level counter | Embedded line flags | Embedded line flags |
| Synchronization trade-off | Amortized barriers/fences over slices | Fine-grained readiness, high flag overhead | Finer-grained readiness with lower flag overhead |
| Typical motivation | High useful bandwidth for larger transfers | Low latency for small transfers | Latency/bandwidth compromise on supported paths |
| True direct primitives in this snapshot | Supported where enabled | Direct API emulated by nondirect primitives | Direct API emulated by nondirect primitives |

These fractions describe protocol payload representation, not a measured link-efficiency ranking. Simple's barriers, fences, counter accesses, channel scheduling, transport staging, and payload memory traffic can outweigh its representation advantage for small messages.

The tuner uses topology-sensitive estimates, protocol enable filters, and correction factors. Its choice cannot be inferred from payload fraction alone.

## 16. Configuration and performance considerations

Relevant controls in this snapshot include:

| Control | Purpose / caution |
|---|---|
| `NCCL_PROTO=Simple` | Restrict collective protocol choice; does not force Ring or a transport |
| `NCCL_ALGO=Ring` | Useful with the previous setting for an isolated Ring/Simple experiment |
| `NCCL_BUFFSIZE` | Set Simple buffer bytes; changes step/slice capacities and memory footprint |
| `NCCL_NTHREADS` | Affect Ring/Tree worker-thread budgets; synchronization overhead threads are added later |
| `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS` | Bound channel construction; final choices also depend on config/topology and enqueue tuning |
| `NCCL_P2P_READ_ENABLE` | Override P2P read versus write preference |
| `NCCL_P2P_DIRECT_DISABLE` | Disable the ordinary same-process direct-pointer transport choice |
| `NCCL_P2P_PCI_CHUNKSIZE`, `NCCL_P2P_NVL_CHUNKSIZE`, `NCCL_P2P_NET_CHUNKSIZE` | Point-to-point chunk controls, not Ring collective chunk controls |

The default Ring/Simple worker budget is chosen as 256 or up to 512 based on the topology's modeled intra-node bandwidth. Enqueueing then adds a synchronization warp for Ring/Simple; the primitive can exclude that warp from `nworkers`. Thus a kernel's total thread count should not be confused with `NCCL_SIMPLE_MAX_NTHREADS=512`.

For smaller messages, enqueueing reduces channels or worker threads based on thresholds, before adding Simple's synchronization overhead. More channels are not always faster: they reduce data per channel while increasing synchronization, memory, and resource pressure.

The central tuning trade-offs are:

- **Larger slices:** amortize fences and posting, but delay readiness and can reduce pipeline responsiveness.
- **Smaller slices:** improve forwarding latency, but increase synchronization frequency.
- **More workers:** increase memory-level parallelism, but consume registers and CTA resources.
- **More channels:** expose independent pipelines, but increase connection storage and contention.
- **Direct writes:** eliminate receive copies where available, but still require rendezvous and correct visibility.
- **Direct reads:** change remote access direction; benefit depends on interconnect and memory access behavior.

This snapshot does not expose an Intel-style UC/WT/ST cache-policy sweep through these Simple controls. Its PTX access semantics cannot be read as a recommendation to use a particular CRI cache hint.

## 17. Comparison with this repository's CRI `simple_pcie`

### 17.1 The names conceal a different protocol design

In this workspace, `simple_pcie` selects `Rt64_128_PCIE` with `SequentialTransmit`; the CRI `simple` and `simple_pcie_p` names select that same message protocol with `ParallelTransmit`.

For CRI SIMD16:

```text
message bytes per lane = 16
transaction bytes = 16 lanes * 16 bytes = 256
payload lanes = 16 * 15 / 16 = 15
useful payload bytes = 15 * 16 = 240
payload fraction = 240 / 256 = 93.75%
```

`insertFlags` places readiness flags into the message, and `recvMessages` loads the message and checks flags at multiple lane positions. This is an embedded-flag approach, conceptually closer to LL128's data/flag organization than to NCCL Simple's separate counters. That comparison is architectural, not a claim of identical packet layout, atomicity assumptions, or performance.

Local source references: [message constants](rt64_128.hpp#L3), [flag insertion](rt64_128.hpp#L104), [message polling](rt64_128.hpp#L448), [algorithm dispatch](allreduce.cpp#L1199), [cache policies](cache_control.hpp#L1), and [sequential transport](sequential_transmit.hpp#L25).

| Dimension | NCCL ordinary Simple | Current CRI `simple_pcie` |
|---|---|---|
| Payload readiness | Separate `tail` counter | Flags embedded in each message |
| Ordinary buffering | Eight logical FIFO steps | SequentialTransmit message-ring layout |
| Main publication granularity | Slice | 256-byte SIMD16 message |
| Payload representation | No per-line flags | 240 payload bytes per 256 wire bytes |
| Visibility implementation | PTX fences, counter stores, volatile loads | Intel LSC operations and configured cache controls |
| Payload direct-buffer variants | Explicit push/pull primitive paths | Not established by the NCCL study |

The table does not imply that SequentialTransmit lacks all synchronization or buffering; it identifies the different readiness/publication mechanism.

### 17.2 Earlier CRI results, kept separate from NCCL findings

The earlier test used two CRI GPUs on `10.99.62.220`, Intel MPI, `simple_pcie`, SIMD16, UC stores, and `-g 64 -w 32`. Element counts are binary multiples of 1,048,576; BF16 elements occupy two bytes. All four correctness checks passed. Times are medians of three maximum-rank kernel measurements.

| Elements per rank | Useful bytes per rank | Kernel time | Useful payload goodput |
|---|---:|---:|---:|
| 1M | 2 MiB | 0.130208 ms | 16.106 GB/s |
| 2M | 4 MiB | 0.203229 ms | 20.638 GB/s |
| 4M | 8 MiB | 0.381771 ms | 21.973 GB/s |
| 32M | 64 MiB | 2.972917 ms | 22.573 GB/s |

Goodput here is:

```text
G = useful input bytes per rank / maximum-rank kernel time
```

For a Ring AllReduce with `P` ranks, the usual useful-data bus-bandwidth normalization is:

```text
normalizedBusBandwidth = G * 2 * (P - 1) / P
```

For two ranks this factor is one. Neither number directly measures PCIe wire throughput: embedded flags, counter traffic, packet headers, loads, stores, and implementation-specific transfers are not accounted for in `G`. Multiplying by two to sum both ranks gives aggregate rank throughput, not one direction's physical-link bandwidth.

The CRI payload representation's ideal flag-removal gain is only `256/240 ≈ 1.0667` if everything else is unchanged. This is not a prediction that an NCCL-style Simple port would improve measured goodput by 6.67%: its fences, slice sizes, credits, and memory-access pattern would also change.

Raw earlier logs remain in `/tmp/cri-simple-pcie-results.WkM31A` locally and `/home/caozhong/ipc_simple_pcie_qMTAXe` on the CRI host. Those temporary paths are convenience pointers, not required for reproducing the pinned NCCL source study.

## 18. Porting lessons, correctness risks, and an experiment plan

### 18.1 What is worth borrowing

A useful experimental design for CRI would separate bulk payload transfer from control publication:

```text
producer waits for credit
workers write an unflagged bulk slice
workers join the posting role
platform-correct cross-device publication ordering
producer publishes the completed sequence
consumer observes the sequence with platform-correct visibility
workers load/reduce/copy the payload
workers join the credit-return role
consumer returns credit
```

The practical opportunities are fewer payload polling loads, no per-message flag insertion, and amortized control operations. Keep the existing embedded-flag protocol as a baseline rather than replacing it before the new contract is validated.

### 18.2 Preconditions that need independent proof on CRI

1. **Publication ordering:** payload stores from all workers must become visible before the counter can be observed remotely. A local barrier alone does not establish this across devices.
2. **Receive visibility:** observing a fresh counter must not permit stale payload reads; counter and payload cache policies both matter.
3. **Counter accesses:** shared/imported addresses must support the chosen aligned counter load/store width, visibility, and polling semantics. Do not assume CUDA/PTX system scope maps directly to SYCL scopes across separate devices.
4. **Credit ordering:** the producer must not overwrite data until every consumer worker has finished reading it.
5. **Progress:** communicating workgroups and posting roles must be schedulable without relying on unsupported independent forward progress.
6. **Persistent state:** operation transitions, chunk rounding, tail partitions, and empty slices must keep producer and consumer positions aligned.
7. **Memory placement:** distinguish imported device memory from host fallback; different placement changes both correctness assumptions and performance.
8. **Cache policy:** earlier UC/WT/ST observations for the existing protocol do not prove the same behavior for a new bulk-plus-counter protocol. Retest every chosen access/fence combination.

### 18.3 Failure modes to test deliberately

| Mistake | Likely symptom |
|---|---|
| Publish `tail` before all payload writes are ordered | Intermittent incorrect data |
| Poll a counter or payload through stale caching | Hang or stale data after apparent readiness |
| Return `head` before workers finish reading | Corruption when the FIFO wraps |
| Advance wait and post counters by different amounts | Permanent credit/readiness wait |
| Skip zero-byte slices or final partitions | Hangs at message tails or next operation |
| Reuse a direct pointer slot before acceptance | Wrong buffer address or rendezvous hang |
| Remove a seemingly unused direct constructor handshake | Peer constructor waits forever |
| Assume network request completion implies GPU visibility | Stale receive data on GDR paths |
| Reserve too many workers/channels for available residency | Progress/resource stalls or reduced bandwidth |

### 18.4 Suggested staged validation

This is a proposed experiment, not work performed by this study:

1. Implement one producer/consumer bulk FIFO with two ranks, one channel, explicit credits, and UC as the initial control/payload baseline where supported.
2. Validate publication/reuse with many more iterations than the FIFO depth, sequence-specific payload patterns, deliberate producer/consumer delays, and consecutive operations without resetting counters.
3. Exercise zero elements, one element, vector-alignment boundaries, odd partitions, partial final slices, exact FIFO wrap boundaries, and large messages.
4. Compare slice sizes such as 16 KiB, 64 KiB, 256 KiB, and 1 MiB while keeping all other parameters fixed. These are test choices, not NCCL requirements.
5. Add fused reduction/forwarding, then multiple channels/workgroups, then any direct output-buffer optimization.
6. Re-run the existing 1M, 2M, 4M, and 32M cases, correctness first, followed by repeated maximum-rank timings and a dispersion statistic.
7. Isolate host fallback, device P2P push, and any validated pull mode. Report useful GB/s separately from estimated protocol-byte traffic and measured hardware counters.
8. Sweep UC/WT/ST only after the ordering contract is established, with bounded timeouts and fresh processes for configurations that hang.

For a genuine NCCL comparison, use an NVIDIA CUDA machine and a benchmark linked against this pinned NCCL build. Select `NCCL_PROTO=Simple` and `NCCL_ALGO=Ring`, record topology and transport selection, and compare the same useful byte sizes. NCCL's CUDA kernels are not directly runnable on the CRI cards.

## 19. Bottom line

The defining idea of NCCL Simple is **“bulk slice first, publish readiness separately, reclaim through credits.”** Its throughput strategy combines that representation with worker/posting-role overlap, vectorized fused reduction/copy, direct-buffer shortcuts, and transport-aware completion rules.

The most useful lesson for this CRI repository is not a particular packet width or cache hint. It is the separation of data movement from progress publication, together with the need to prove ordering, visibility, reuse, and forward progress on the target platform. The existing `simple_pcie` name describes a different embedded-flag protocol and should not be treated as evidence that NCCL Simple has already been implemented.

## Pinned source references

[version]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/makefiles/version.mk#L2
[allreduce-api]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/all_reduce.cc#L12
[protocols]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/primitives.h#L24
[conn-info]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/include/devcomm.h#L85
[channel-peers]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/include/devcomm.h#L271
[control-mem]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/include/comm.h#L36
[simple-wait]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/prims_simple.h#L120
[simple-op]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/prims_simple.h#L176
[simple-connections]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/prims_simple.h#L352
[simple-roles]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/prims_simple.h#L438
[simple-direct]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/prims_simple.h#L497
[reduce-copy]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/common_kernel.h#L183
[memory-helpers]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/op128.h#L236
[ring-allreduce]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/all_reduce.h#L12
[tree-allreduce]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/all_reduce.h#L170
[buffer-defaults]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/init.cc#L492
[algo-selection]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/enqueue.cc#L1155
[host-chunks]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/enqueue.cc#L1309
[p2p-setup]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/p2p.cc#L328
[p2p-connect]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/p2p.cc#L445
[shm-connect]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/shm.cc#L146
[shm-progress]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/shm.cc#L285
[net-send]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/net.cc#L931
[net-recv]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/transport/net.cc#L1108
[sendrecv]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/collectives/device/sendrecv.h#L12
[tuning]: https://github.com/CaoZhongZ/nccl/blob/d97a32fac84f69cbd022ffdc57d5687c760742c7/src/graph/tuning.cc#L113
