from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(labels, preds):
    mse = mean_squared_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)
    mae = mean_absolute_error(labels, preds)

    return {"mse": mse, "rmse": rmse, "mae": mae}