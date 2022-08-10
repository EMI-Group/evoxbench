from django.contrib import admin

import nasbench201


@admin.register(nasbench201.models.NASBench201Result)
class NASBench201Admin(admin.ModelAdmin):
    list_display = ["id", "index", "phenotype"]
    search_fields = ["phenotype"]
# Register your models here.
