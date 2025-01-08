import pandas as pd
import seaborn as sns
from faicons import icon_svg
from shiny import reactive
from shiny.express import input, render, ui
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load train and test metrics
cwd = os.getcwd()
data_path = os.path.join(cwd, "epoch_data.csv")
df = pd.read_csv(data_path)

# Load test and validation results
validation_results_path = os.path.join(cwd, "validation_results.csv")
test_results_path = os.path.join(cwd, "test_results.csv")
validation_results = pd.read_csv(validation_results_path)
test_results = pd.read_csv(test_results_path)

# Set up the Shiny UI 
ui.page_opts(title="Training Metrics Dashboard", fillable=True)

# Create a sidebar for filter controls
with ui.sidebar(title="Filter controls"):
    # Input slider to filter data by epoch range
    ui.input_slider(
        "epoch", "Epoch", 
        int(df["epoch"].min()),  # Minimum epoch value from the dataset
        int(df["epoch"].max()),  # Maximum epoch value from the dataset
        int(df["epoch"].max())   # Default slider position (max epoch)
    )

    # Display the loss for the selected epoch
    @render.text
    def epoch_loss():
        selected_epoch = input.epoch()
        loss = df.loc[df["epoch"] == selected_epoch, "loss"].values
        return f"Loss at Epoch {selected_epoch}: {loss[0] if len(loss) > 0 else 'N/A'}"

# Define the main dashboard layout
with ui.layout_column_wrap(fill=False):
    # Display a value box for total epochs
    with ui.value_box(showcase=icon_svg("chart-line")):
        "Total Epochs"

        @render.text
        def total_epochs():
            # Render the total number of epochs from the dataset
            return f"{df['epoch'].max()}"

    # Display a value box for maximum validation accuracy
    with ui.value_box(showcase=icon_svg("chart-bar")):
        "Max Validation Accuracy"

        @render.text
        def max_val_accuracy():
            # Render the maximum validation accuracy from the dataset
            return f"{df['validation_accuracy'].max():.2f}"

    # Display a value box for minimum loss
    with ui.value_box(showcase=icon_svg("chart-area")):
        "Min Loss"

        @render.text
        def min_loss():
            # Render the minimum loss value from the dataset
            return f"{df['loss'].min():.2f}"

# Create the layout for plots
with ui.layout_columns():
    # Card for Loss vs Epoch plot
    with ui.card(full_screen=True):
        ui.card_header("Loss vs Epoch")

        @render.plot
        def loss_plot():
            # Get the filtered dataset based on the selected epoch
            filtered = filtered_df()
            # Create a scatter plot for loss vs epoch
            plot = sns.scatterplot(
                data=filtered,
                x="epoch",
                y="loss",
                marker="o",
            )
            plot.set_title("Loss vs Epoch")
            return plot

    # Card for Validation Accuracy vs Epoch plot
    with ui.card(full_screen=True):
        ui.card_header("Validation Accuracy vs Epoch")

        @render.plot
        def val_accuracy_plot():
            # Get the filtered dataset based on the selected epoch
            filtered = filtered_df()
            # Create a scatter plot for validation accuracy vs epoch
            plot = sns.scatterplot(
                data=filtered,
                x="epoch",
                y="validation_accuracy",
                marker="o",
            )
            plot.set_title("Validation Accuracy vs Epoch")
            return plot

# Add confusion matrix plots for validation and test results
with ui.layout_columns():
    # Card for Validation Confusion Matrix
    with ui.card(full_screen=True):
        ui.card_header("Validation Confusion Matrix")

        @render.plot
        def validation_confusion_matrix():
            true_labels = validation_results["True Label"]
            predicted_labels = validation_results["Predicted Label"]
            cm = confusion_matrix(true_labels, predicted_labels, labels=["Helix", "Sheet", "Coil"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Helix", "Sheet", "Coil"], yticklabels=["Helix", "Sheet", "Coil"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Validation Confusion Matrix")
            return fig

    # Card for Test Confusion Matrix
    with ui.card(full_screen=True):
        ui.card_header("Test Confusion Matrix")

        @render.plot
        def test_confusion_matrix():
            true_labels = test_results["True Label"]
            predicted_labels = test_results["Predicted Label"]
            cm = confusion_matrix(true_labels, predicted_labels, labels=["Helix", "Sheet", "Coil"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Helix", "Sheet", "Coil"], yticklabels=["Helix", "Sheet", "Coil"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Test Confusion Matrix")
            return fig

# Include custom CSS for additional styling
ui.include_css("styles.css")

# Reactive calculation to filter the dataset by the selected epoch range
@reactive.calc
def filtered_df():
    # Filter the dataset to include rows where the epoch is less than or equal to the selected epoch
    filt_df = df[df["epoch"] <= input.epoch()]
    return filt_df
