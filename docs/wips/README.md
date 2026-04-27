# WIP-0: Wabizabi Improvement Proposals

**Author:** [Esten](https://youtu.be/_zMiWGnKWZw?si=zxS7xwc4CV-3i_X3)\
**Created:** 17-Apr-2026

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [Introduction](#introduction)
- [Index](#index)
- [How WIPs Work](#how-wips-work)
  - [Numbering and filenames](#numbering-and-filenames)
  - [Required metadata](#required-metadata)
  - [Type](#type)
  - [Status](#status)
  - [Topics](#topics)

</details>

## Introduction

This document is the index for the Wabizabi Improvement Proposals, which I abbreviate to WIPs.

A WIP is two things at once. It is a proposal, in that I'm putting forward a position on how some part of Wabizabi should be designed and inviting people to push back. It is also a design document, in that once a proposal has been worked through, the WIP is what stays behind to explain what the design is and why it took the shape it did.

The format follows [Python's PEPs](https://peps.python.org/pep-0001/) and [Apache Flink's FLIPs](https://cwiki.apache.org/confluence/display/Flink/Flink+Improvement+Proposals); I borrowed liberally from both. This approach is heavily inspired by Mario Zechner's philosophy on building systems. My earlier iterations of Wabizabi degraded into a mess because I was building without a clear mental model. [As Zechner points out](https://youtu.be/RjfbvDXpFls?si=n-EVsns6wGaedUHF&t=1016), forcing yourself to write things down creates friction, and that friction is what builds an understanding of the system in your head. These documents serve as that friction.

This document also describes how WIPs are written, numbered, and tracked.

## Index
 
|  | WIP | Title | Author |
| :--- | :--- | :--- | :--- |
| AD | [1](./wip-0001.md) | [The Wabizabi Architecture](./wip-0001.md) | Me |
| AP | [2](./wip-0002.md) | [*Core Primitives*](./wip-0002.md) | Me |
| AP | [3](./wip-0003.md) | [*Trace, Replay, and Durability*](./wip-0003.md) | Me |
| AP | [4](./wip-0004.md) | [*The Authoring API*](./wip-0004.md) | Me |
| AP | [5](./wip-0005.md) | [*Effects, Handlers, and Middleware*](,/wip-0005.md) | Me |
| AP | [6](./wip-0006.md) | [*Content and Modalities*](./wip-0006.md) | Me |
| AP | [7](./wip-0007.md) | [*Standard Library Modules*](./wip-0007.md) | Me |
| AP | [8](./wip-0008.md) | [*Wazi: The Reference CLI and TUI*](./wip-0008.md) | Me |

## How WIPs Work
 
### Numbering and filenames

WIPs are numbered monotonically, starting at 0001 (this index is 0000). The filename is `wip-NNNN.md`, where `NNNN` is the zero-padded number. Numbers are not reused. A retired or replaced WIP keeps its number, with its status updated accordingly.

### Required metadata

Each WIP carries a small metadata block at the top: title, author, status, type, and creation date.

### Type

Every WIP is one of three types. The type identifies what kind of change the document is proposing.

- **Architecture** (*A*); a specification of core logic or structural design in the Wabizabi ecosystem. WIPs 1 through 8 are all of this kind.
- **Feature** (*F*); a proposal for an addition or enhancement that doesn't change the underlying architecture.
- **Process** (*P*); a guideline for how the project itself is developed: standards, conventions, workflow.

### Status

Every WIP carries one of six statuses. The status describes where the document is in its lifecycle.

- **Planned** (*P*); the WIP is on the roadmap but the document hasn't been written, or is still being authored privately.
- **Draft** (*D*); the document is published and open for feedback. Its design is still expected to change.
- **Accepted** (*A*); the design has been worked through and is ready to be implemented. The document is no longer expected to change in substance.
- **Baked** (*B*); the design has been implemented and merged into the codebase. The document now describes what is, not what is proposed.
- **Superseded** (*S*); a later WIP has replaced this one. The document is preserved for historical context, with a pointer to its successor.
- **Ditched** (*X*); the proposal was set aside without being implemented. The document is preserved so the reasoning is not lost.

### Topics

I'm leaving topic categories empty for now. I'd rather wait until enough WIPs exist to suggest natural groupings than invent categories before I know what they should be.
