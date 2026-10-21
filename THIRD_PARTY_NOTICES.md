# Third-party notices

Emula is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).
It uses the following third-party components.

| Component | Use | License | Copyright | Source |
|---|---|---|---|---|
| Maia-3 | The model weights (converted to Core ML by `Tools/maia3_export/`, not committed) and the encoding, decoding and sampling logic ported from its source | AGPL-3.0 | CSSLab, University of Toronto | https://github.com/CSSLab/maia3, https://huggingface.co/UofTCSSLab |
| Scacchi | Chess rules: legal moves, replaying games. Our own bridge to shakmaty (© Niklas Fiekas and contributors); its linked crates are listed in Scacchi's `THIRD_PARTY_LICENSES.txt` | GPL-3.0-or-later | The Scacchi authors | https://github.com/fianchettolab/scacchi |
| python-chess | The export, fixtures and parity tools (not linked into the package) | GPL-3.0-or-later | © Niklas Fiekas and contributors | https://github.com/niklasf/python-chess |

Apps that ship Emula should show Maia-3's attribution and citation, and the notices of
Scacchi.

## Maia-3 citation

Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson.
*Chessformer: A Unified Architecture for Chess Modeling.* The Fourteenth International
Conference on Learning Representations (ICLR 2026). arXiv:2605.19091.
