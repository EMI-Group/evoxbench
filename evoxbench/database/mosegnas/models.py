from operator import mod
from nasbenchbase.models import NASBenchResult
from django.db import models


class MoSegNASResult(NASBenchResult):
    # TODO
    flops = models.FloatField(default=-1)
    # params = models.FloatField(default=-1)
    # final_test_accuracy = models.JSONField(default=dict)
    # final_validation_accuracy = models.JSONField(default=dict)
