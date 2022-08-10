from email.policy import default
from nasbenchbase.models import NASBenchResult
from django.db import models


class NASBench201Result(NASBenchResult):
    index = models.IntegerField(db_index=True)
    phenotype = models.CharField(max_length=256, db_index=True)
    cost12 = models.JSONField(default=dict)
    cost200 = models.JSONField(default=dict)
    # res12 = models.JSONField(default=dict)
    # res200 = models.JSONField(default=dict)
    hw_info = models.JSONField(default=dict)
    more_info = models.JSONField(default=dict)
