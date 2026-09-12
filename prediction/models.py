from django.db import models
from django.contrib.auth.models import User


class Feedback(models.Model):

    name = models.CharField(max_length=100)

    message = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Prediction(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    patient_name = models.CharField(
        max_length=100
    )

    age = models.IntegerField()

    gender = models.CharField(
        max_length=20
    )

    weight = models.FloatField()

    height = models.FloatField()

    bmi = models.FloatField()

    prediction = models.IntegerField()

    probability = models.FloatField()

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.patient_name