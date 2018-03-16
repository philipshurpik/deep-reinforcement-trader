import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def render_sticks(ax, quotes, width=1000, colorup='#00FF00', colordown='#FF0000', alpha=0.8):
    for q in quotes:
        t, open, high, low, close = q[:5]
        timestamp = dates.date2num(t)

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(timestamp, timestamp), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )
        rect = Rectangle(
            xy=(timestamp, lower),
            width=0.5/len(quotes),
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)
        rect.set_linewidth((width * 0.8)/len(quotes))
        ax.add_line(vline)
        ax.add_patch(rect)


def render(values):
    fig, ax = plt.subplots(figsize=(20, 8))
    render_sticks(ax, values, width=20*50)
    ax.autoscale_view()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=40)
    plt.grid()
    plt.show()
