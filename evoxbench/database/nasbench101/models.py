from operator import mod
from nasbenchbase.models import NASBenchResult
from django.db import models


class NASBench101Result(NASBenchResult):
    # epoch4 = models.JSONField(default=dict)
    # epoch12 = models.JSONField(default=dict)
    # epoch36 = models.JSONField(default=dict)
    # epoch108 = models.JSONField(default=dict)
    # image = models.ImageField(upload_to='', null=True)
    flops = models.FloatField(default=-1)
    params = models.FloatField(default=-1)
    final_test_accuracy = models.JSONField(default=dict)
    final_validation_accuracy = models.JSONField(default=dict)
