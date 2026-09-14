from django.contrib import admin
from django.urls import path
from prediction.views import (
    home,
    register,
    user_login,
    user_logout,
    feedback,
    prediction_history,
    prediction_detail,
    delete_prediction,
)
urlpatterns = [
    path("admin/", admin.site.urls),

    # Register page
    path("", home, name="home"),
    path("register/", register, name="register"),

    # Patient prediction page
    path("prediction/", home, name="prediction"),
    path("login/", user_login, name="login"),
    path("logout/", user_logout, name="logout"),
    path("feedback/", feedback, name="feedback"),
    path("history/", prediction_history, name="history"),
    path(
    "history/<int:prediction_id>/",
    prediction_detail,
    name="prediction_detail"
),
path(
    "history/<int:prediction_id>/delete/",
    delete_prediction,
    name="delete_prediction"
),
]