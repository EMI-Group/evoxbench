from operator import mod
from nasbenchbase.models import NASBenchResult
from django.db import models


class MoSegNASResult(NASBenchResult):
    params = models.BigIntegerField(default=-1)
    flops = models.BigIntegerField(default=-1)
    latency = models.FloatField(default=-1)
    FPS = models.FloatField(default=-1)
    mIoU = models.FloatField(default=-1)
