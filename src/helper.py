import plotly.graph_objects as go
import plotly.express as px


def plot_loss(results):
    """This function plots the training and validation loss.
    results: keras training history as pandas df
    loss: Training loss
    val_loss: validation loss
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["loss"],
            name="Training Loss",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["val_loss"],
            name="Validation Loss",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis=dict(title="Epochs"),
        yaxis=dict(title="Loss"),
    )
    fig.show()


def plot_accuracy(results):
    """This function plots the training and validation acuuracy.
    results: keras training history as pandas df
    accuracy: training accuracy
    val_accuracy: validation accuracy
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["accuracy"],
            name="Training Accuracy",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["val_accuracy"],
            name="Validation Accuracy",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_layout(
        title="Training vs Validation Accuracy",
        xaxis=dict(title="Epochs"),
        yaxis=dict(title="Accuracy"),
    )
    fig.show()
