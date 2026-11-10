# Third party notices

This repository packages local experiments built on two MIT-licensed projects.

| Project | Upstream revision | Retained notice |
| --- | --- | --- |
| Battery-Sim-Agent, copyright 2026 Opqrst | `354f2ddcbb8d14d6dfe9673aac55199976d2d42c` | [MIT license](experiments/battery/provenance/UPSTREAM_LICENSE) |
| OPAL, copyright 2026 Xingjian Bai, Jia-Hao Liang, Jayden Teoh, Vincent Sitzmann | `8b16759a3e9930916ed582e28068267859f935f8` | [MIT license](experiments/opal/LICENSE.upstream) |

The battery compatibility patch and experiment runners adapt Battery-Sim-Agent. The OPAL experiment uses its MNIST example recipe and generated updater programs. Original copyright and permission notices remain applicable to those portions. New packaging and plotting code is covered by the root MIT license.

PyBaMM, PyTorch, torchvision, NumPy, Matplotlib, Ax, and other dependencies retain their own licenses. They are installed separately. MNIST and Ollama model weights are not redistributed here; users obtain them from their respective sources under the applicable terms.
