from django.contrib import admin

import natsbenchsss


@admin.register(natsbenchsss.models.NATSBenchResult)
class NATSBenchResultAdmin(admin.ModelAdmin):
    list_display = ["id", "index", "phenotype"]
    search_fields = ["phenotype"]
# Register your models here.
