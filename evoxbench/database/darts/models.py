from nasbenchbase.models import NASBenchResult
from django.db import models


class NASBench301Result(NASBenchResult):
    normal = models.CharField(max_length=128)
    normal_concat = models.CharField(max_length=128)
    reduce = models.CharField(max_length=128)
    reduce_concat = models.CharField(max_length=128)
    epochs = models.IntegerField()
    dataset_id = models.IntegerField()
    dataset_path = models.CharField(max_length=256)
