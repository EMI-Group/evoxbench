from django.contrib import admin

import mosegnas


@admin.register(mosegnas.models.MoSegNASResult)
class MoSegNASAdmin(admin.ModelAdmin):
    list_display = ["id", "phenotype"]
    search_fields = ["phenotype"]
