from django.contrib import admin

import nasbench101


@admin.register(nasbench101.models.NASBench101Result)
class NASBench101Admin(admin.ModelAdmin):
    list_display = ["id", "index"]
    search_fields = ["index"]
# Register your models here.
