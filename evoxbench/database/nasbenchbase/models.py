from django.db import models


class NASBenchResult(models.Model):
    """
    Base class of nas bench results
    """
    id = models.BigAutoField(primary_key=True)
    index = models.TextField(max_length=256, db_index=True, null=True, blank=False)
    phenotype = models.JSONField(default=dict)
    genotype = models.JSONField(default=dict)
    result = models.JSONField(default=dict)

    class Meta:
        abstract = True
