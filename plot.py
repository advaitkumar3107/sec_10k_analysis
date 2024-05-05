import plotly.graph_objects as go


def make_plots(fig, df, col, subplot_idx, title_text, legend_x, legend_y):
    """
    Input : 
    fig (plotly.subplot) : The figure which needs to be populated with data
    df (pd.DataFrame) : The data to be plotted
    col (int): The position of the plot (row)
    subplot_idx (int): The position of the plot in the row
    title_text (str) : The title of the subplot
    legend_x (float) : The relative x-coordinate of the legend
    legend_y (float) : The relative y-coordinate of the legend
    
    Output : 
    fig (plotly.subplot) : The figure after being populated with data
    """
    
    bar_width = 0.2  ## width of bar in each group
    num_cols = len(df.columns)
    group_width = num_cols * bar_width  ## width of each group

    for i, column in enumerate(df.columns):
        #### Add the bars in the figure
        fig.add_trace(
            go.Bar(
                x=[x + i * bar_width for x in range(len(df))],
                y=df[column],
                name=column,
                text=[f"{val:.2f}" for val in df[column]],
                textposition='outside',
                hoverinfo='text',
                hovertext=[f"{val:.2f}" for val in df[column]],
                width=bar_width,
                legendgroup=f'group{subplot_idx}',  # Assign legend group
                showlegend=True  # Show legend for each trace
            ), 
            row=1, col=subplot_idx
        )
    
    # Update subplot's x-axis
    fig.update_xaxes(
        tickvals=[x + group_width / 2 - bar_width / 2 for x in range(len(df))],
        ticktext=df.index,
        row=1, col=subplot_idx
    )

    # Customize the layout and position the legend
    fig.update_layout(
        title_text=title_text,
        barmode='group',
        showlegend=True,
        legend=dict(x=legend_x, y=legend_y, orientation="h")
    )
    return fig