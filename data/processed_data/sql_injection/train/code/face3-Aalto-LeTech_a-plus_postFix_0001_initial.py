# -*- coding: utf-8 -*-


from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelWithInheritance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                <fix/>('content_type', models.ForeignKey(editable=False, to='contenttypes.ContentType', null=True, on_delete=models.CASCADE)),</fix>
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
    ]
