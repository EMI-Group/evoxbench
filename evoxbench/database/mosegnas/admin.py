from django.contrib import admin

import mosegnas


@admin.register(mosegnas.models.MoSegNASResult)
class MoSegNASAdmin(admin.ModelAdmin):
    # TODO
    list_display = ["id", "index"]
    search_fields = ["index"]
# Register your models here.
