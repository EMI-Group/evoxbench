# Neural Architecture Search as Multiobjective Optimization Benchmarks: Problem Formulation and Performance Assessment [[arXiv]](https://arxiv.org/abs/2208.04321)

## Preparation Steps

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

## Database

Visit this webpage for more information: https://github.com/liuxukun2000/evoxdatabase

## Acknowledgement

Codes are developed upon: [NAS-Bench-101](https://github.com/google-research/nasbench)
, [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201), [NAS-Bench-301](https://github.com/automl/nasbench301)
, [NATS-Bench](https://xuanyidong.com/assets/projects/NATS-Bench)
, [Once for All](https://github.com/mit-han-lab/once-for-all)
, [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [Django](https://www.djangoproject.com/)
, [pymoo](https://pymoo.org/)

## Support

If you have any question, you can contract us through the QQ group (ID: 297969717).

<img src="qq_group.jpg" alt="drawing" width="30%"/>
