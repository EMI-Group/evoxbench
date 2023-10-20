<h1 align="center">
  <img src=./_static/evox_logo.png alt="Logo" height="24em"/>
  <strong>EvoXBench</strong>
  <br>
  Neural Architecture Search as Multiobjective Optimization Benchmarks: Problem Formulation and Performance Assessment <a href=https://arxiv.org/abs/2208.04321>arXiv</a>
  <p align="center">
  ❤️ Found EvoX helpful? Please consider giving it a star to show your support! ⭐
</p>
</h1>


🌟 In the **ever-evolving domain** of deep learning and computer vision, the pursuit for the pinnacle of **network architecture design** is paramount. The advent of **Neural Architecture Search (NAS)** signals a paradigm shift, automating design intricacies for heightened accuracy. Yet, as the gamut of deep learning applications broadens, the clamor for **versatile network architectures** that cater to multifaceted design criteria surges. Welcome to **EvoXBench** — our trailblazing framework poised to metamorphose NAS endeavors into **holistic multi-objective optimization challenges**, heralding a fresh epoch for research via **evolutionary multiobjective optimization (EMO)** algorithms.

---

## 📢 Latest News & Updates

- 📌 We're thrilled to announce that EvoXBench has been updated to version **1.0.3**! This latest release addresses bugs in IN-1KMOP5, IN-1KMOP6, and the NB201 benchmark.

  We urge all users to transition to this latest version of EvoXBench. If you're already onboard with EvoXBench, give this command a spin: `pip install evoxbench==1.0.3`.

  Your trust in EvoXBench means the world to us! For any queries or feedback, our doors are always open.

---


## ⭐️ Key Features

### 📐 General NAS Problem Formulation
- Cast NAS tasks into the mold of generalized multi-objective optimization problems.
- Undertake an intricate exploration of NAS's nuanced traits through the prism of optimization.

### 🛠️ Efficient Benchmarking Pipeline
- Presenting an end-to-end conduit, primed for proficient benchmark assessments of EMO algorithms.
- Shed the shackles of GPUs or bulky frameworks like PyTorch/TensorFlow, championing far-reaching compatibility.

### 📊 Comprehensive Test Suites
- Encompassing a wide spectrum of datasets and search spaces, and a trio of hardware devices.
- Navigate challenges graced with up to eight objectives for a comprehensive evaluation escapade.

---

## 🎬 Get Started

<p align="center">
  <a href="https://www.emigroup.tech/wp-content/uploads/2023/02/tutorial.mp4">
    <img src="https://github.com/EMI-Group/evoxbench/blob/main/assets/video%20cover.png" alt="Dive into the tutorial" width="450"/>
  </a>
  <br>
  <small>Tap the image to embark on the introductory video voyage.</small>
</p>

**Note:** Embarking on some problems? Be informed of certain nuances regarding IGD computations. For the nitty-gritty, our documentation is your best friend.

---

## 🛠 Setup & Installation

1. 📥 Download requisite files:
    - ``database.zip`` 
      via [Google Drive](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing)
      or [Baidu云盘（提取码：mhgs）](https://pan.baidu.com/s/1PwWloA543-81O-GFkA7GKg)

    - ``data.zip``
      via [Google Drive](https://drive.google.com/file/d/1fUZtpTjfEQao2unLKaspL8fOq4xdSXt2/view?usp=sharing)
      or [Baidu云盘（提取码：lfib）](https://pan.baidu.com/s/1yopkISKyjbWIHXFV_Op3pg)

2. 💻 Run `pip install evoxbench` to get the benchmark.

3. 🖥 Configure the benchmark:

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

---

## 🗃 About the Database

Explore our comprehensive database and understand its structure and content. Check it out [here](https://github.com/liuxukun2000/evoxdatabase).

---

## 👥 Community & Support

- **Issues & Queries**: Use the issue tracker for bugs or questions.
- **Contribute**: Submit your enhancements through a pull request (PR).
- **Join our Community**: We have an active **QQ group** (ID: 297969717). Come join us! 
  
---

## 🙌 Credits & Acknowledgements

A big shoutout to the following projects that have made EvoXBench possible:

 [NAS-Bench-101](https://github.com/google-research/nasbench)
 [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
 [NAS-Bench-301](https://github.com/automl/nasbench301)
 [NATS-Bench](https://xuanyidong.com/assets/projects/NATS-Bench)
 [Once for All](https://github.com/mit-han-lab/once-for-all)
 [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer)
 [Django](https://www.djangoproject.com/)
 [pymoo](https://pymoo.org/)
