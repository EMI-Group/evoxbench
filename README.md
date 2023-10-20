<h1 align="center">
  <img src=./_static/evox_logo.png alt="Logo" height="24em"/>
  <strong>EvoXBench</strong>
  <br>
  Neural Architecture Search as Multiobjective Optimization Benchmarks: Problem Formulation and Performance Assessment <a href=https://arxiv.org/abs/2208.04321>arXiv</a>
</h1>

---

## 📢 Latest News & Updates

🚀 **Version 1.0.3 is Out Now!** We've patched some bugs in benchmarks and improved the overall stability. It's recommended for all users to upgrade.

pip install evoxbench==1.0.3

---

## 🎬 Get Started with EvoXBench

<p align="center">
  <a href="https://www.emigroup.tech/wp-content/uploads/2023/02/tutorial.mp4">
    <img src="https://github.com/EMI-Group/evoxbench/blob/main/assets/video%20cover.png" alt="Watch the tutorial" width="450"/>
  </a>
  <br>
  <small>Click on the image to watch the introduction video.</small>
</p>

**Note:** Certain problems have specific limitations regarding IGD calculations. Please refer to the documentation for details.

---

## 🛠 Setup & Installation

1. Download the following two requried files:
    - ``database.zip`` file
      from [Google Drive](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing)
      or [Baidu云盘（提取码：mhgs）](https://pan.baidu.com/s/1PwWloA543-81O-GFkA7GKg)

    - ``data.zip`` file
      from [Google Drive](https://drive.google.com/file/d/1fUZtpTjfEQao2unLKaspL8fOq4xdSXt2/view?usp=sharing)

      or [Baidu云盘（提取码：lfib）](https://pan.baidu.com/s/1yopkISKyjbWIHXFV_Op3pg)

2. ``pip install evoxbench`` to install the benchmark.

3. Configure the benchmark via the following steps:

```python
    from evoxbench.database.init import config

    config("Path to databae", "Path to data")
    # For example
    # If you have the following structure
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
    # Then you should do:
    # config("/home/Downloads/database", "/home/Downloads/data")
```

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

