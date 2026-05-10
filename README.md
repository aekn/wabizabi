# Wabizabi

Wabizabi is a runtime for orchestrating AI systems in Python. I'm trying to position it a layer below the abstractions most current frameworks expose, with the aim of being unopinionated about what an agent looks like, what modalities are in play, and where the system runs. The hope is that production concerns like durability, replay, observability, and cost control are properties of the runtime rather than after thoughts.

This is a monorepo. It contains two packages: **wabizabi**, the runtime, and **wazi**, its reference command-line tool and terminal interface.

## Status

I'm rewriting the project against a new architecture. The earlier iteration is preserved under `_legacy/` directories so I can refer back to it, but it isn't the design I'm building toward. New development starts from WIP-1 onward.

## Design

The architecture is being worked out in a series of Wabizabi Improvement Proposals. The format follows the conventions of [Python's PEPs](https://peps.python.org/pep-0001/) and [Apache Flink's FLIPs](https://cwiki.apache.org/confluence/display/Flink/Flink+Improvement+Proposals). The index lives in [docs/wips/](./docs/wips/README.md).
