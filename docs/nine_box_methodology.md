# 9-Box Grid Methodology

## Overview

The 9-Box Grid (also known as the McKinsey talent matrix or GE-McKinsey matrix) is a widely adopted talent management framework used by HR professionals and organisational leaders to assess employees along two dimensions simultaneously: **current performance** and **future potential**.

The result is a 3x3 matrix that categorises each employee into one of nine boxes, providing a structured basis for talent reviews, succession planning, and targeted development investments.

---

## The 9-Box Matrix

The grid maps **Performance** (horizontal axis) against **Potential** (vertical axis), with each axis divided into three levels: Low, Medium, and High.

```
                        PERFORMANCE
              Low          Medium         High
         ┌────────────┬────────────┬────────────┐
   High  │ Inconsist- │   High     │            │
         │ ent Player │ Potential  │   Star     │
         │    (7)     │    (4)     │    (1)     │
P        ├────────────┼────────────┼────────────┤
O Medium │ Develop-   │   Core     │   High     │
T        │ ment       │  Player    │ Performer  │
E        │ Needed (8) │    (5)     │    (2)     │
N        ├────────────┼────────────┼────────────┤
T  Low   │   Under    │  Average   │   Solid    │
I        │ Performer  │ Performer  │ Performer  │
A        │    (9)     │    (6)     │    (3)     │
L        └────────────┴────────────┴────────────┘
```

---

## Box Definitions

### Box 1 -- Star (High Performance, High Potential)

Top talent who consistently exceed expectations and demonstrate strong growth capacity. These employees are future leaders and key retention priorities.

**Recommendation**: Invest heavily in retention. Offer stretch assignments, executive mentoring, and leadership fast-track programmes.

### Box 2 -- High Performer (High Performance, Medium Potential)

Consistently strong contributors who may have moderate but not exceptional growth trajectory. They are reliable top performers and team anchors.

**Recommendation**: Recognise contributions publicly. Provide cross-functional exposure and explore potential-unlocking coaching.

### Box 3 -- Solid Performer (High Performance, Low Potential)

Excellent at their current role but unlikely to advance significantly. They are specialists and subject-matter experts who provide stability.

**Recommendation**: Maintain engagement with meaningful work. Explore lateral moves to broaden experience and reignite growth.

### Box 4 -- High Potential (Medium Performance, High Potential)

Employees with significant growth capacity whose current performance has room for improvement. Often newer employees or those in transitional roles.

**Recommendation**: Accelerate development with targeted training, challenging projects, and a performance improvement plan with clear milestones.

### Box 5 -- Core Player (Medium Performance, Medium Potential)

The backbone of the organisation. Solid contributors who meet expectations and have moderate growth potential. Typically the largest group.

**Recommendation**: Encourage skill deepening. Pair with mentors and set incremental performance goals to move towards higher boxes.

### Box 6 -- Average Performer (Medium Performance, Low Potential)

Employees who meet basic expectations but show limited growth indicators. May be plateaued or disengaged.

**Recommendation**: Identify specific skill gaps. Provide structured training and quarterly check-ins to track improvement.

### Box 7 -- Inconsistent Player (Low Performance, High Potential)

High potential individuals whose performance is below expectations. This gap may indicate poor role fit, lack of engagement, or external factors.

**Recommendation**: Investigate root causes of inconsistency. Provide coaching and a focused performance improvement plan with short-term targets.

### Box 8 -- Development Needed (Low Performance, Medium Potential)

Employees who underperform but show some growth indicators. With the right support, they may move to higher boxes.

**Recommendation**: Assign a dedicated mentor. Create a 90-day development plan with measurable outcomes and regular feedback sessions.

### Box 9 -- Under Performer (Low Performance, Low Potential)

Employees who consistently underperform with limited growth capacity. Requires immediate intervention.

**Recommendation**: Initiate a formal performance improvement plan. Provide clear expectations, weekly reviews, and consider role realignment.

---

## Threshold Configuration

The classification boundaries determine how performance and potential scores map to Low, Medium, and High tiers.

### Default Thresholds

| Axis | Low | Medium | High |
|---|---|---|---|
| Performance | < 2.5 | 2.5 to < 3.5 | >= 3.5 |
| Potential | < 2.5 | 2.5 to < 3.5 | >= 3.5 |

### Custom Thresholds

Thresholds are configurable via the `NineBoxClassifier` constructor:

```python
classifier = NineBoxClassifier(
    perf_thresholds=(2.0, 4.0),  # More lenient low / stricter high
    pot_thresholds=(2.5, 3.5),   # Standard potential boundaries
)
```

Adjusting thresholds changes the distribution of employees across boxes. Organisations should calibrate thresholds to their specific context:

- **Strict thresholds** (e.g., 3.0, 4.0): fewer Stars and High Performers, more granular differentiation at the top.
- **Lenient thresholds** (e.g., 2.0, 3.0): larger top-right quadrant, useful when overall scores trend lower.

---

## Use Cases in HR

### Succession Planning

The 9-Box Grid identifies a pipeline of future leaders by highlighting **Stars** (Box 1) and **High Potentials** (Box 4) as candidates for succession tracks. These employees receive accelerated development and are prepared for critical roles before vacancies occur.

### Talent Reviews

During calibration meetings, leadership teams use the grid to align on employee assessments, reduce rater bias, and ensure consistent evaluation standards across departments. The visual matrix facilitates structured discussion about each employee's placement and trajectory.

### Development Investment Allocation

The grid guides where to direct limited development budgets:

- **Boxes 1, 2, 4**: High-priority investment (retention bonuses, leadership programmes, stretch assignments).
- **Boxes 3, 5, 6**: Maintenance investment (skill deepening, lateral moves, mentorship).
- **Boxes 7, 8, 9**: Remediation investment (performance improvement plans, coaching, role reassignment).

### Compensation and Rewards

Organisations often tie compensation decisions to grid placement, ensuring top talent receives competitive pay, equity grants, and recognition, while underperformers receive performance-linked adjustments.

### Workforce Planning

Aggregate 9-Box distributions reveal organisational health. A healthy distribution typically shows a bell curve centred around Box 5, with a healthy representation in Boxes 1--4 and a manageable proportion in Boxes 7--9.

---

## Limitations and Ethical Considerations

### Subjectivity in Ratings

Performance and potential scores are often based on subjective manager assessments. Implicit bias related to gender, ethnicity, age, or personality can distort placements. Organisations should complement manager ratings with objective metrics (goals met, peer reviews, project outcomes).

### Potential Is Difficult to Measure

Unlike performance (which can be observed), potential is inherently speculative. It often correlates with visibility, confidence, and political capital rather than true capability. Over-reliance on potential scores can perpetuate biases that favour extroverted or well-networked employees.

### Static Snapshot Problem

The 9-Box Grid captures a point-in-time assessment and does not reflect trajectory. An employee moving from Box 9 to Box 5 over two periods shows significant improvement that a single snapshot would miss. Always consider trends alongside current placement.

### Label Stigma

Labels like "Under Performer" can become self-fulfilling prophecies if they are communicated carelessly or used punitively. The grid should be a development tool, not a ranking mechanism. Employees should understand that placement is not permanent and that support is available.

### Lack of Context

The grid does not account for external factors such as workload distribution, team dynamics, organisational changes, or personal circumstances. A drop in performance may reflect systemic issues rather than individual capability.

### Fairness Auditing

When using ML-derived scores to populate the 9-Box Grid, organisations should:

- Audit score distributions for demographic disparities (gender, ethnicity, age).
- Test for disparate impact using the four-fifths rule or statistical parity metrics.
- Apply causal inference methods (as implemented in this platform) to distinguish genuine effects from confounded associations.
- Regularly recalibrate thresholds to ensure equitable outcomes.

### Recommended Safeguards

1. **Multi-source feedback**: Combine manager, peer, self, and objective metrics.
2. **Regular recalibration**: Review thresholds and distributions quarterly.
3. **Transparency**: Communicate the methodology and criteria to all employees.
4. **Appeals process**: Allow employees to challenge their placement with evidence.
5. **Trend analysis**: Track movement across periods, not just current placement.
