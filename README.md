<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/bench-logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/bench-logo-light.png">
    <img alt="EvoXBench Logo" height="48em" src="./assets/bench-logo-light.png">
  </picture>
  <br>
  Neural Architecture Search as Multiobjective Optimization Benchmarks: Problem Formulation and Performance Assessment <a href=https://arxiv.org/abs/2208.04321>arXiv</a>
</h1>

---

EvoXBench is a platfrom offering **instant benchmarking** of evolutionary multi-objective optimization (EMO) algorithms in neural architecture search (NAS), with ready to use test suites. It facilitates efficient performance assessments **with NO requirement of GPUs or PyTorch/TensorFlow**, enhancing accessibility for a broader range of research applications. It encompasses extensive test suites that cover a variety of datasets (CIFAR10, ImageNet, etc.), search spaces (NASBench101, NASBench201, NATS, DARTS, ResNet50, Transformer, MNV3, etc.), and hardware devices (Eyeriss, GPUs, Samsung Note10, etc.). It provides a versatile interface **compatible with multiple programming languages** (Java, Matlab, Python, etc.).

---


## 📢 Latest News & Updates

- EvoXBench has been updated to version **1.0.3**! This latest release addresses bugs in IN-1KMOP5, IN-1KMOP6, and the NB201 benchmark.

  If you're already onboard with EvoXBench, give this command a spin: `pip install evoxbench==1.0.3`.



## ⭐️ Key Features

### 📐 General NAS Problem Formulation
- Formulating NAS tasks into general multi-objective optimization problems.
- Exploring NAS's nuanced traits through the prism of evolutionary optimization.

### 🛠️ Efficient Benchmarking Pipeline
- Presenting an end-to-end worflow for instant benchmark assessments of EMO algorithms.
- Providing instant fitness evaluations as numerical optimization.

### 📊 Comprehensive Test Suites
- Encompassing a wide spectrum of datasets, search spaces, and hardware devices.
- Ready-to-use test multi-objective optimization suites with up to eight objectives.


## Get Started

<p align="center">
  <a href="https://www.emigroup.tech/wp-content/uploads/2023/02/tutorial.mp4">
    <img src="https://github.com/EMI-Group/evoxbench/blob/main/assets/video%20cover.png" alt="Dive into the tutorial" width="450"/>
  </a>
  <br>
  <small>Tap the image to embark on the introductory video voyage.</small>
</p>



## Setup & Installation

1. Download requisite files:
    - ``database.zip``
      via [Google Drive](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing)
      or [Baidu云盘（提取码：mhgs）](https://pan.baidu.com/s/1PwWloA543-81O-GFkA7GKg)

    - ``data.zip``
      via [Google Drive](https://drive.google.com/file/d/1fUZtpTjfEQao2unLKaspL8fOq4xdSXt2/view?usp=sharing)
      or [Baidu云盘（提取码：lfib）](https://pan.baidu.com/s/1yopkISKyjbWIHXFV_Op3pg)

2. Run `pip install evoxbench` to get the benchmark.

3. Configure the benchmark:

```python
    from evoxbench.database.init import config

    config("Path to database", "Path to data")
    # For instance:
    # With this structure:
    # /home/Downloads/
    # └─ database/
    # |  |  __init__.py
    # |  |  db.sqlite3
    # |  |  ...
    # |
    # └─ data/
    #    └─ darts/
    #    └─ mnv3/
    #    └─ ...
    # Then, execute:
    # config("/home/Downloads/database", "/home/Downloads/data")
```

## About the Database

Explore our comprehensive database and understand its structure and content. Check it out [here](https://github.com/liuxukun2000/evoxdatabase).

## Community & Support

- Use the issue tracker for bugs or questions.
- Submit your enhancements through a pull request (PR).
- We have an active QQ group (ID: 297969717).
- Official Website: https://evox.group/

## Sister Projects
- EvoX: A computing framework for distributed GPU-aceleration of evolutionary computation, supporting a wide spectrum of evolutionary algorithms and test problems. Check out [here](https://github.com/EMI-Group/evox).


## Citing EvoXBench

If you use EvoXBench in your research and want to cite it in your work, please use:
```
@article{EvoXBench,
  title={Neural Architecture Search as Multiobjective Optimization Benchmarks: Problem Formulation and Performance Assessment},
  author={Lu, Zhichao and Cheng, Ran and Jin, Yaochu and Tan, Kay Chen and Deb, Kalyanmoy},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements

A big shoutout to the following projects that have made EvoXBench possible:

 [NAS-Bench-101](https://github.com/google-research/nasbench),
 [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201),
 [NAS-Bench-301](https://github.com/automl/nasbench301),
 [NATS-Bench](https://xuanyidong.com/assets/projects/NATS-Bench),
 [Once for All](https://github.com/mit-han-lab/once-for-all),
 [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer),
 [Django](https://www.djangoproject.com/),
 [pymoo](https://pymoo.org/).








