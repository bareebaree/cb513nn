import pandas as pd
import seaborn as sns
from faicons import icon_svg
from shiny import reactive
from shiny.express import input, render, ui
import os
# Load train and test metrics

cwd = os.getcwd()

data_path = os.path.join(cwd,"epoch_data.csv")
df = pd.read_csv(data_path)

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

# Include custom CSS for additional styling
ui.include_css("styles.css")

# Reactive calculation to filter the dataset by the selected epoch range
@reactive.calc
def filtered_df():
    # Filter the dataset to include rows where the epoch is less than or equal to the selected epoch
    filt_df = df[df["epoch"] <= input.epoch()]
    return filt_df
