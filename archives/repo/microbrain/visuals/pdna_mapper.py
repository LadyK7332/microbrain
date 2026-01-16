import matplotlib.pyplot as plt
import numpy as np
import yaml


def hex_plot(pdna_file):
    data = yaml.safe_load(open(pdna_file))
    traits = {
        "power": data["drives"]["power"],
        "maintenance": data["drives"]["maintenance"],
        "civilization": data["drives"]["civilization"],
        "novelty": data.get("novelty_dampener", 0.5),
        "risk": data.get("risk_tolerance", 0.5),
    }
    labels, vals = list(traits.keys()), list(traits.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.plot(angles, vals, "o-", linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(data.get("id", "unknown"), pad=20)
    plt.savefig(f"visuals/output/{data.get('id','pdna')}.png", bbox_inches="tight")
    plt.close(fig)
