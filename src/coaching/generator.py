def generate_summary(positive, negative):

    summary = []

    # Negative coaching
    for phrase, _ in negative:

        if "called" in phrase.lower():
            summary.append(
                f"Customer frustration detected: \"{phrase}\""
            )

        elif "transfer" in phrase.lower():
            summary.append(
                f"Multiple transfers may have impacted experience: \"{phrase}\""
            )

        elif "frustrating" in phrase.lower():
            summary.append(
                f"Customer dissatisfaction increased: \"{phrase}\""
            )

        else:
            summary.append(
                f"Potential issue detected: \"{phrase}\""
            )


    # Positive coaching
    for phrase, _ in positive:

        if "sorry" in phrase.lower():
            summary.append(
                f"Agent showed empathy: \"{phrase}\""
            )

        elif "understand" in phrase.lower():
            summary.append(
                f"Agent acknowledged customer concern: \"{phrase}\""
            )

        else:
            summary.append(
                f"Positive interaction: \"{phrase}\""
            )


    return summary