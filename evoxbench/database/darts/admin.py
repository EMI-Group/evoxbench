from django.contrib import admin

import darts


@admin.register(darts.models.NASBench301Result)
class NASBench301Admin(admin.ModelAdmin):
    list_display = ["id", "genotype"]
    search_fields = ["genotype"]
# Register your models here.
