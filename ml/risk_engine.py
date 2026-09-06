def classify_risk(ttf):

    if ttf > 6:
        return "NORMAL"

    elif ttf > 3:
        return "WATCH"

    elif ttf > 1:
        return "HIGH RISK"

    else:
        return "CRITICAL"


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":
    test_values = [
        8.5,
        5.2,
        2.4,
        0.7
    ]

    for ttf in test_values:
        risk = classify_risk(ttf)
        print(f"TTF: {ttf:.2f} h → {risk}")
