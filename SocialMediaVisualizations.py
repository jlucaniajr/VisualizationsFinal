# SocialMediaVisualizations.py
# Complete stitched file with mental health choropleths, social media bar charts, and heatmaps

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set working directory
os.chdir('/Users/johnjrlucania/Desktop/Visualizations/VisualizationsFinal')

# Load datasets
mental_health = pd.read_csv('data/MentalHealthSurvey/Mental_Health_Survey_Feb_20_22.csv')
census_regions = pd.read_csv('data/USCensusBureau_RegionsAndDivisions.csv')
smmh = pd.read_csv('data/smmh.csv')

# --- Mental Health Choropleth Map ---
mental_health_clean = mental_health.iloc[2:].copy()
mental_health_clean.columns = mental_health.iloc[0]
score_map = {"Not at all": 0, "Several days": 1, "More than half of the days": 2, "Nearly every day": 3}
likert_cols = mental_health_clean.columns[mental_health_clean.isin(score_map.keys()).any()].tolist()
phq9_cols = likert_cols[:9]
gad7_cols = likert_cols[9:]
for col in phq9_cols + gad7_cols:
    mental_health_clean[col] = mental_health_clean[col].map(score_map)
mental_health_clean['GAD7_Score'] = mental_health_clean[gad7_cols].sum(axis=1)
mental_health_clean['PHQ9_Score'] = mental_health_clean[phq9_cols].sum(axis=1)
mental_health_clean['GAD7_Negative'] = mental_health_clean['GAD7_Score'] >= 10
mental_health_clean['PHQ9_Negative'] = mental_health_clean['PHQ9_Score'] >= 10
state_col = 'In which state do you live? - State'
state_summary = mental_health_clean.groupby(state_col).agg(
    Total_Respondents=('GAD7_Score', 'count'),
    GAD7_Negative_Count=('GAD7_Negative', 'sum'),
    PHQ9_Negative_Count=('PHQ9_Negative', 'sum')
).reset_index()
state_summary['GAD7_Negative_Percent'] = (state_summary['GAD7_Negative_Count'] / state_summary['Total_Respondents']) * 100
state_summary['PHQ9_Negative_Percent'] = (state_summary['PHQ9_Negative_Count'] / state_summary['Total_Respondents']) * 100
choropleth_data = pd.merge(state_summary, census_regions, left_on=state_col, right_on='State', how='left')
fig_map = go.Figure()
fig_map.add_trace(go.Choropleth(
    locations=choropleth_data['State Code'], 
    z=choropleth_data['GAD7_Negative_Percent'],
    locationmode='USA-states', 
    colorscale='Reds', 
    colorbar_title="Anxiety (%)", 
    visible=False,  # Changed from True to False
    text=choropleth_data['State'],
    hovertemplate='%{text}<br>Anxiety: %{z:.2f}%<extra></extra>'
))
fig_map.add_trace(go.Choropleth(
    locations=choropleth_data['State Code'], 
    z=choropleth_data['PHQ9_Negative_Percent'],
    locationmode='USA-states', 
    colorscale='Blues', 
    colorbar_title="Depression (%)", 
    visible=True,  # Changed from False to True
    text=choropleth_data['State'],
    hovertemplate='%{text}<br>Depression: %{z:.2f}%<extra></extra>'
))

fig_map.update_layout(title_text='Mental Health Negative Responses by State', geo=dict(scope='usa', projection=dict(type='albers usa'), showcountries=False, showlakes=True, lakecolor='white'), dragmode=False)

# --- Social Media Bar Chart ---
platform_col = '7. What social media platforms do you commonly use?'
time_col = '8. What is the average time you spend on social media every day?'
smmh_clean = smmh[[time_col, platform_col]].dropna()
smmh_expanded = smmh_clean.assign(platform=smmh_clean[platform_col].str.split(', ')).explode('platform')
total_counts = smmh_clean.groupby(time_col).size().reset_index(name='total')
platform_counts = smmh_expanded.groupby([time_col, 'platform']).size().reset_index(name='count')
platform_counts = platform_counts.merge(total_counts, on=time_col)
platform_totals = smmh_expanded.groupby(time_col).size().reset_index(name='total_platform_mentions')
platform_counts = platform_counts.merge(platform_totals, on=time_col)
platform_counts['scaled'] = (platform_counts['count'] / platform_counts['total_platform_mentions']) * platform_counts['total']
platforms = platform_counts['platform'].unique()
platform_colors = px.colors.qualitative.Dark24
fig_social = go.Figure()
fig_social.add_trace(go.Bar(x=total_counts[time_col], y=total_counts['total'], marker_color='lightgray', name='Total Responses', hoverinfo='skip'))
for idx, platform in enumerate(platforms):
    data = platform_counts[platform_counts['platform'] == platform]
    fig_social.add_trace(go.Bar(
        x=data[time_col],
        y=data['scaled'],
        name=platform,
        marker_color=platform_colors[idx % len(platform_colors)],
        visible='legendonly',
        hovertemplate='%{x}<br>' + platform + ': %{y:.2f}%<extra></extra>',
    ))
fig_social.update_layout(barmode='overlay', bargap=0, title='Social Media Platform Breakdown by Daily Usage Time (Raw Counts)', xaxis_title='Daily Social Media Usage Time', yaxis_title='Number of Responses', showlegend=False, xaxis=dict(categoryorder='array', categoryarray=['Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours', 'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours']))

# --- Mental Health vs. Platform Heatmap ---
depression_col = '18. How often do you feel depressed or down?'
anxiety_col = '13. On a scale of 1 to 5, how much are you bothered by worries?'
smmh_filtered = smmh[[platform_col, depression_col, anxiety_col]].dropna()
smmh_filtered[depression_col] = smmh_filtered[depression_col].astype(int)
smmh_filtered[anxiety_col] = smmh_filtered[anxiety_col].astype(int)
smmh_expanded = smmh_filtered.assign(platform=smmh_filtered[platform_col].str.split(', ')).explode('platform')
def make_pivot(df, score_col):
    counts = df.groupby([score_col, 'platform']).size().reset_index(name='count')
    pivot_count = counts.pivot(index=score_col, columns='platform', values='count').fillna(0)
    pivot_percent = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100
    return pivot_count, pivot_percent
depr_counts, depr_percent = make_pivot(smmh_expanded, depression_col)
anx_counts, anx_percent = make_pivot(smmh_expanded, anxiety_col)
fig_heatmap = go.Figure()
fig_heatmap.add_trace(go.Heatmap(z=depr_counts.values, x=depr_counts.columns, y=depr_counts.index, colorscale='Blues', colorbar_title='Count', visible=True, hovertemplate='Platform: %{x}<br>Severity: %{y}<br>Count: %{z}<extra></extra>'))
fig_heatmap.add_trace(go.Heatmap(z=anx_counts.values, x=anx_counts.columns, y=anx_counts.index, colorscale='Reds', colorbar_title='Count', visible=False, hovertemplate='Platform: %{x}<br>Severity: %{y}<br>Count: %{z}<extra></extra>'))
fig_heatmap.add_trace(go.Heatmap(z=depr_percent.values, x=depr_percent.columns, y=depr_percent.index, colorscale='Blues', colorbar_title='Percent', visible=False, hovertemplate='Platform: %{x}<br>Severity: %{y}<br>Percent: %{z:.2f}%<extra></extra>'))
fig_heatmap.add_trace(go.Heatmap(z=anx_percent.values, x=anx_percent.columns, y=anx_percent.index, colorscale='Reds', colorbar_title='Percent', visible=False, hovertemplate='Platform: %{x}<br>Severity: %{y}<br>Percent: %{z:.2f}%<extra></extra>'))
fig_heatmap.update_layout(title='Mental Health Severity vs Social Media Platform', xaxis_title='Platform', yaxis_title='Severity (1 = Low, 5 = High)', margin=dict(l=40, r=40, t=80, b=40), plot_bgcolor='white')

# Merge time_col with anxiety/depression levels
smmh_scores = smmh[[time_col, depression_col, anxiety_col]].dropna().copy()
smmh_scores[depression_col] = smmh_scores[depression_col].astype(int)
smmh_scores[anxiety_col] = smmh_scores[anxiety_col].astype(int)

# Define negative thresholds
smmh_scores['GAD7_Negative'] = smmh_scores[anxiety_col] >= 4  # Anxiety: 4+
smmh_scores['PHQ9_Negative'] = smmh_scores[depression_col] >= 3  # Depression: 3+

# Group by time bins
conclusion_summary = smmh_scores.groupby(time_col).agg(
    Total_Respondents=('GAD7_Negative', 'count'),
    GAD7_Negative=('GAD7_Negative', 'sum'),
    PHQ9_Negative=('PHQ9_Negative', 'sum')
).reset_index()

# Calculate percentages
conclusion_summary['Anxiety_Rate'] = (conclusion_summary['GAD7_Negative'] / conclusion_summary['Total_Respondents']) * 100
conclusion_summary['Depression_Rate'] = (conclusion_summary['PHQ9_Negative'] / conclusion_summary['Total_Respondents']) * 100

# Order time bins consistently
time_order = ['Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours',
              'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours']
conclusion_summary[time_col] = pd.Categorical(conclusion_summary[time_col], categories=time_order, ordered=True)
conclusion_summary = conclusion_summary.sort_values(time_col)

# Scatter plot instead of bar
fig_conclusion = go.Figure()
fig_conclusion.add_trace(go.Scatter(
    x=conclusion_summary[time_col], y=conclusion_summary['Anxiety_Rate'],
    mode='markers+lines', name='Anxiety (GAD-7 ≥ 4)', marker=dict(color='crimson', size=10),
    line=dict(color='crimson'), hovertemplate='%{x}<br>Anxiety: %{y:.2f}%<extra></extra>'
))
fig_conclusion.add_trace(go.Scatter(
    x=conclusion_summary[time_col], y=conclusion_summary['Depression_Rate'],
    mode='markers+lines', name='Depression (PHQ-9 ≥ 3)', marker=dict(color='steelblue', size=10),
    line=dict(color='steelblue'), hovertemplate='%{x}<br>Depression: %{y:.2f}%<extra></extra>'
))
fig_conclusion.update_layout(title='Rates of Anxiety and Depression by Social Media Usage Time',
    xaxis_title='Daily Social Media Usage Time', yaxis_title='Percentage of Respondents (%)',
    margin=dict(l=40, r=40, t=80, b=40), plot_bgcolor='white')


# --- Generate Integrated HTML ---
with open("jLucaniaMentalHealth.html", "w") as f:
    f.write("""<html><head><title>Mental Health and Social Media</title>
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
    </head><body><div class="progress-container"><div class="progress-bar"></div></div><div class="header">
  <a href="#introduction">Introduction</a>
  <a href="#mental-health">Mental Health</a>
  <a href="#social-media">Social Media Usage</a>
  <a href="#heatmap">Comparison & Correlation</a>
  <a href="#conclusion">Conclusion</a>
</div>
""")

    # Introduction
    f.write("""<div class="section" id="introduction"><h1>Introduction</h1><p>This report investigates the intersection between mental health indicators and social media usage patterns among U.S. adults. Leveraging survey data on anxiety (GAD-7), depression (PHQ-9), and daily social media engagement across multiple platforms, the analysis aims to reveal underlying trends and correlations. Specifically, the report visualizes mental health prevalence geographically, tracks platform-specific social media usage over time, and explores the correlation between usage intensity and mental health outcomes.
</p></div>""")

    # Mental Health Choropleth Map
    f.write("""<div class="section" id="mental-health"><h2>Mental Health (Anxiety, Depression)</h2><p>Toggle map view: <span class="platform-toggle" data-map-trace="0">Anxiety (GAD-7)</span>, <span class="platform-toggle" data-map-trace="1">Depression (PHQ-9)</span></p><div class="plot-container">""")
    f.write(fig_map.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("""</div><p>Both the anxiety (GAD-7) and depression (PHQ-9) maps reveal overlapping patterns, but with key differences in distribution. Anxiety rates tend to be more concentrated, with states like Idaho, North Dakota, Mississippi, and West Virginia standing out as higher prevalence areas. These clusters may reflect factors such as rural isolation, economic strain, or limited access to mental health services. Depression, on the other hand, appears more broadly distributed across regions, with elevated levels not only in the same high-anxiety states but also extending further into the Midwest and East Coast. This suggests that while both conditions share common underlying drivers, depression may be more consistently influenced by broader socioeconomic disparities. Together, these maps highlight both targeted areas for anxiety-specific interventions and the need for expanded mental health resources addressing depression on a wider scale.
</p></div>""")

    # Social Media Usage
    f.write("""<div class="section" id="social-media"><h2>Social Media Usage Over Time</h2><p>Toggle platforms (By percent of total):""")
    clickable_platforms = ', '.join([f'<span class="platform-toggle" data-platform="{platform}">{platform}</span>' for platform in platforms])
    f.write(clickable_platforms)
    f.write("""</p><div class="plot-container">""")
    f.write(fig_social.to_html(full_html=False, include_plotlyjs=False))
    f.write("""</div><p>The social media usage breakdown highlights clear differences in platform engagement across daily time ranges. <span class=\"platform-toggle\" data-platform=\"YouTube\">YouTube</span>, <span class=\"platform-toggle\" data-platform=\"Instagram\">Instagram</span>, <span class=\"platform-toggle\" data-platform=\"Facebook\">Facebook</span>, and <span class=\"platform-toggle\" data-platform=\"Discord\">Discord</span> show consistent usage across all time bins, with <strong>YouTube</strong> slightly stronger among heavy users, reflecting its flexible content length and broad appeal. In contrast, <span class=\"platform-toggle\" data-platform=\"TikTok\">TikTok</span> and <span class=\"platform-toggle\" data-platform=\"Reddit\">Reddit</span> skew heavily toward higher usage brackets, particularly among those spending more than five hours per day on social media. These platforms are designed for immersive, extended engagement, which aligns with their dominance among heavy users. <span class=\"platform-toggle\" data-platform=\"Snapchat\">Snapchat</span> and <span class=\"platform-toggle\" data-platform=\"Discord\">Discord</span> maintain steady engagement across moderate to high usage ranges, serving as routine communication tools. Meanwhile, <span class=\"platform-toggle\" data-platform=\"Pinterest\">Pinterest</span> and <span class=\"platform-toggle\" data-platform=\"Twitter\">Twitter</span> show lower overall engagement, concentrated mainly in lighter usage groups.
</p></div>""")

    # Mental Health Heatmap
    f.write("""<div class="section" id="heatmap"><h2>Mental Health Severity by Platform</h2>
    <p><strong>Select Mental Health Measure:</strong> <span class="platform-toggle" data-heatmap-type="depression">Depression</span>, <span class="platform-toggle" data-heatmap-type="anxiety">Anxiety</span></p>
    <p><strong>Select Data View:</strong> <span class="platform-toggle" data-heatmap-mode="counts">Counts</span>, <span class="platform-toggle" data-heatmap-mode="percent">Percentages</span></p>
    <div class="plot-container">""")
    f.write(fig_heatmap.to_html(full_html=False, include_plotlyjs=False))
    f.write("""</div><p>This heatmap illustrates the relationship between mental health severity (Depression, Anxiety) and social media platform usage. Severity levels range from 1 (low) to 5 (high), offering a clear visual representation of mental health distributions across different platforms. The counts view highlights raw engagement numbers, while the percentages view reveals relative proportions within each severity level.

The heatmaps reveal that Facebook, YouTube, and Instagram consistently exhibit higher mental health severity rates for both depression and anxiety, particularly at mid-to-high severity levels. These findings suggest that more frequent engagement on these platforms may be associated with elevated mental health risks. In contrast, TikTok, Reddit, and Pinterest show lower associations, indicating less overlap with severe mental health symptoms in this dataset.
</p></div>""")

    f.write("""<div class="section" id="conclusion"><h2>Conclusion: Mental Health Trends by Social Media Usage</h2>
    <p>As a final analysis, this scatter plot summarizes the relationship between daily social media usage and rates of negative mental health outcomes (anxiety and depression). The data shows a clear upward trend for both conditions:
</p>
    <ul>
        <li><strong>Depression rates</strong> (blue) start at around 25% for those spending less than an hour per day but climb to nearly 90% among users who spend more than five hours online.
</li>
        <li><strong>Anxiety rates</strong> (red) follows a similar pattern, rising from around 26% to over 70% as time spent increases.
</li>
            </ul>
    <div class="plot-container">""")
    f.write(fig_conclusion.to_html(full_html=False, include_plotlyjs=False))
    f.write("""</div><p>Both trends suggest that increased social media use is associated with heightened mental health risks. Depression consistently remains higher across all time groups but shows a sharper increase after 2-3 hours of daily use. Anxiety rises more gradually but still peaks alongside heavy usage.</p>""")
    f.write("""<p>The scatter plot demonstrates a strong correlation between daily social media usage time and the prevalence of anxiety and depression symptoms. Both conditions intensify as usage time increases, with depression rates consistently higher across all groups. This trend suggests that longer periods of social media engagement may be a contributing factor to worsening mental health, particularly for depression.</p></div>""")
    f.write("""</div></div>""")


    # JavaScript
    f.write("""<script>window.addEventListener('load', () => {
        // Scroll progress bar
        window.addEventListener('scroll', () => {
            const scrollTop = window.scrollY;
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrolled = (scrollTop / scrollHeight) * 100;
            document.querySelector('.progress-bar').style.width = scrolled + '%';
        });

        const mapPlot = document.querySelectorAll('.plotly-graph-div')[0];
        const socialPlot = document.querySelectorAll('.plotly-graph-div')[1];
        const heatmapPlot = document.querySelectorAll('.plotly-graph-div')[2];

        // Map toggle
        document.querySelectorAll('[data-map-trace]').forEach(el => {
            el.addEventListener('click', () => {
                const traceIndex = parseInt(el.getAttribute('data-map-trace'));
                mapPlot.data.forEach((trace, i) => {
                    Plotly.restyle(mapPlot, {'visible': [i === traceIndex]}, [i]);
                });
            });
        });

        // Social media toggle
        document.querySelectorAll('[data-platform]').forEach(el => {
            el.addEventListener('click', () => {
                const platformName = el.getAttribute('data-platform');
                socialPlot.data.forEach((trace, i) => {
                    if(trace.name === platformName) {
                        Plotly.restyle(socialPlot, {'visible': [true]}, [i]);
                    } else if(trace.name !== 'Total Responses') {
                        Plotly.restyle(socialPlot, {'visible': 'legendonly'}, [i]);
                    }
                });
            });
        });

        // Heatmap toggle (separated logic)
        let heatmapType = 'depression'; // depression or anxiety
        let heatmapMode = 'counts'; // counts or percent

        function updateHeatmap() {
            const traceMap = {
                'depression_counts': 0,
                'anxiety_counts': 1,
                'depression_percent': 2,
                'anxiety_percent': 3
            };
            const traceIndex = traceMap[heatmapType + '_' + heatmapMode];
            heatmapPlot.data.forEach((trace, i) => {
                Plotly.restyle(heatmapPlot, {'visible': [i === traceIndex]}, [i]);
            });
        }

        document.querySelectorAll('[data-heatmap-type]').forEach(el => {
            el.addEventListener('click', () => {
                heatmapType = el.getAttribute('data-heatmap-type');
                updateHeatmap();
            });
        });
        document.querySelectorAll('[data-heatmap-mode]').forEach(el => {
            el.addEventListener('click', () => {
                heatmapMode = el.getAttribute('data-heatmap-mode');
                updateHeatmap();
            });
        });

        updateHeatmap(); // Initialize
    });</script></body></html>""")

print("HTML report generated successfully.")
