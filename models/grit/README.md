# GRIT inference assets

These are published, pretrained inference artifacts from
[`mrzuang/GRIT_teleop_deploy`](https://github.com/mrzuang/GRIT_teleop_deploy).
This project uses them only for ONNX inference; it does not train, fine-tune,
or resume a checkpoint locally.

## License review required

The source repository currently has no GitHub-detected license. Source links
and integrity hashes establish provenance, not redistribution permission.
Maintainers should confirm the artifact and motion licensing before merging
or redistributing these files. The parent project's MIT license must not be
assumed to relicense these third-party assets.

| File | Source | SHA-256 |
| --- | --- | --- |
| `policy.onnx` | `sim2real/checkpoints/policy.onnx` | `39d573e970b7088275ed68ec8a9597fa9c402d98d5ad211991db7d8d0f5a82b9` |
| `policy.json` | `sim2real/checkpoints/policy.json` | `8a1e5795c898075e4e9263fdff3e85c34747fdb10aa84cacd0af540be7b3dbd4` |
| `kpop_dance.npz` | `sim2real/config/g1/motions/kpop_dance.npz` | `311b56e9a40a55b061d84867fc87028cffe3284616eb6f72eae80f547666de0a` |

The ONNX contract is two inputs—`reference_context [batch, 9, 70]` and
`proprio_history [batch, 990]`—with one `actions [batch, 29]` output.
