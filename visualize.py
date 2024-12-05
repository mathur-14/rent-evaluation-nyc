import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2

# Load the data
df = pd.read_csv('./data/prepared-dataset.csv')

# Initialize the Dash app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1('NYC Real Estate Dashboard'),
    
    # Filters section
    html.Div([
        html.Div([
            html.Label('Available From:'),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=df['availableFrom'].min(),
                max_date_allowed=df['availableFrom'].max(),
                placeholder='Select a date'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Price Range:'),
            dcc.RangeSlider(
                id='price-range',
                min=df['price'].min(),
                max=df['price'].max(),
                step=1000,
                marks={i: f'${i:,}' for i in range(
                    int(df['price'].min()),
                    int(df['price'].max()),
                    int((df['price'].max() - df['price'].min()) / 5)
                )},
                value=[df['price'].min(), df['price'].max()]
            )
        ], style={'width': '100%', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Borough:'),
            dcc.Dropdown(
                id='borough-dropdown',
                options=[{'label': i, 'value': i} for i in df['borough'].unique()],
                multi=True
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Property Type:'),
            dcc.Dropdown(
                id='property-type-dropdown',
                options=[{'label': i, 'value': i} for i in df['propertyType'].unique()],
                multi=True
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Minimum Beds:'),
            dcc.Input(
                id='beds-input',
                type='number',
                min=0,
                step=1
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Minimum Baths:'),
            dcc.Input(
                id='baths-input',
                type='number',
                min=0,
                step=0.5
            )
        ], style={'width': '33%', 'display': 'inline-block'})
    ], style={'padding': '20px', 'backgroundColor': '#f2f2f2'}),
    
    # Map
    html.Div([
        dcc.Graph(id='nyc-map', style={'height': '60vh'})
    ]),
    
    # Property Details Sections
    html.Div([
        # Description Section
        html.Div([
            html.H3('Property Description'),
            html.Div(id='property-description', style={'padding': '10px'})
        ], style={'marginTop': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px'}),
        
        # Police Precinct Section
        html.Div([
            html.H3('Police Precinct Details'),
            html.Div([
                html.Div(id='precinct-details', style={'padding': '10px'})
            ])
        ], style={'marginTop': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px'}),
        
        # Subway Stations Section
        html.Div([
            html.H3('Nearby Subway Stations'),
            html.Div(id='subway-details', style={'padding': '10px'})
        ], style={'marginTop': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px'}),
        
        # Census Data Section
        html.Div([
            html.H3('Census Data'),
            html.Div([
                html.Div([
                    dcc.Graph(id='population-chart', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='race-chart', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='economics-chart', style={'width': '33%', 'display': 'inline-block'})
                ])
            ])
        ], style={'marginTop': '20px', 'backgroundColor': '#f8f9fa', 'padding': '15px'})
    ], id='property-details', style={'display': 'none'})
])

def lat_lon_offset(lat, lon, distance_miles, bearing):
    """
    Calculate new lat/lon given a starting point, distance, and bearing
    
    Args:
        lat: Starting latitude in degrees
        lon: Starting longitude in degrees
        distance_miles: Distance in miles
        bearing: Bearing in degrees (0 = north, 90 = east, etc.)
    
    Returns:
        tuple of (new_lat, new_lon)
    """
    R = 3959  # Earth's radius in miles
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)
    
    # Calculate new latitude
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_miles/R) +
        np.cos(lat_rad) * np.sin(distance_miles/R) * np.cos(bearing_rad)
    )
    
    # Calculate new longitude
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_miles/R) * np.cos(lat_rad),
        np.cos(distance_miles/R) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )
    
    # Convert back to degrees
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return new_lat, new_lon

# Load subway data
subway_df = pd.read_csv('./data/subway.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points in miles using Haversine formula
    """
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# Update existing callback for map
@app.callback(
    Output('nyc-map', 'figure'),
    [Input('date-picker', 'date'),
     Input('price-range', 'value'),
     Input('borough-dropdown', 'value'),
     Input('property-type-dropdown', 'value'),
     Input('beds-input', 'value'),
     Input('baths-input', 'value'),
     Input('nyc-map', 'clickData')]
)
def update_map(date, price_range, boroughs, property_types, min_beds, min_baths, clickData):
    filtered_df = df.copy()
    
    # Apply filters
    if date:
        filtered_df = filtered_df[filtered_df['availableFrom'] >= date]
    
    if price_range:
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) & 
            (filtered_df['price'] <= price_range[1])
        ]
    
    if boroughs:
        filtered_df = filtered_df[filtered_df['borough'].isin(boroughs)]
    
    if property_types:
        filtered_df = filtered_df[filtered_df['propertyType'].isin(property_types)]
    
    if min_beds:
        filtered_df = filtered_df[filtered_df['beds'] >= min_beds]
    
    if min_baths:
        filtered_df = filtered_df[filtered_df['baths'] >= min_baths]
    
    # Create map
    fig = px.scatter_mapbox(
        filtered_df,
        lat='latitude',
        lon='longitude',
        hover_data=['price', 'propertyType', 'beds', 'baths', 'street'],
        color='price',
        size_max=15,
        zoom=10,
        mapbox_style='carto-positron'
    )

    if clickData:
        point = clickData['points'][0]
        prop_lat, prop_lon = point['lat'], point['lon']
        
        # Generate circle points
        circle_points = []
        for bearing in range(0, 361, 10):
            circle_lat, circle_lon = lat_lon_offset(prop_lat, prop_lon, 0.6, bearing)
            circle_points.append([circle_lon, circle_lat])
        
        # Add radius circle
        fig.add_trace(go.Scattermapbox(
            lon=[coord[0] for coord in circle_points],
            lat=[coord[1] for coord in circle_points],
            mode='lines',
            fill='toself',
            fillcolor='rgba(218,31,74,0.3)',
            line=dict(color='rgba(218,31,74,0.3)'),
            hoverinfo='skip'
        ))
        
        # Find and add nearby subway stations
        nearby_stations = []
        for _, station in subway_df.iterrows():
            distance = calculate_distance(
                prop_lat, prop_lon,
                station['stop_lat'], station['stop_lon']
            )
            if distance <= 0.6:
                nearby_stations.append(station)
        
        if nearby_stations:
            nearby_stations_df = pd.DataFrame(nearby_stations)
            # Background circle (larger)
            fig.add_trace(go.Scattermapbox(
                lat=nearby_stations_df['stop_lat'],
                lon=nearby_stations_df['stop_lon'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='black'
                ),
                hoverinfo='skip'
            ))

            # Foreground circle (smaller)
            fig.add_trace(go.Scattermapbox(
                lat=nearby_stations_df['stop_lat'],
                lon=nearby_stations_df['stop_lon'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='yellow'
                ),
                text=nearby_stations_df['stop_name'],
                hoverinfo='text',
                name='Subway Stations'
            ))

    fig.update_layout(
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        mapbox=dict(
            center=dict(
                lat=40.7128,
                lon=-74.0060
            )
        ),
        showlegend=False
    )
    
    return fig

@app.callback(
    [Output('property-details', 'style'),
     Output('property-description', 'children'),
     Output('precinct-details', 'children'),
     Output('subway-details', 'children'),
     Output('population-chart', 'figure'),
     Output('race-chart', 'figure'),
     Output('economics-chart', 'figure')],
    [Input('nyc-map', 'clickData')]
)
def display_property_details(clickData):
    if not clickData:
        return {'display': 'none'}, '', '', '', {}, {}, {}
    
    # Get the clicked property's data
    point_index = clickData['points'][0]['pointIndex']
    selected_property = df.iloc[point_index]
    
    # Description
    description = html.P(selected_property['description'])
    
    # Precinct Details
    precinct_info = html.Div([
        html.P(f"Precinct: {selected_property['Precinct']}"),
        html.P(f"Schools in Precinct: {selected_property['schools_in_precinct']}")
    ])
    
    # Subway Stations
    subway_info = html.P(selected_property['nearby_subway_stations'])
    
    # Population Chart
    population_fig = px.bar(
        x=['Total', 'Male', 'Female'],
        y=[
            selected_property['Total Population'],
            selected_property['Male Population'],
            selected_property['Female Population']
        ],
        title='Population Distribution'
    )
    
    # Race Distribution Chart
    race_data = {
        'Race': ['White', 'Black', 'Asian', 'Hispanic'],
        'Population': [
            selected_property['White Alone'],
            selected_property['Black or African American Alone'],
            selected_property['Asian Alone'],
            selected_property['Hispanic or Latino']
        ]
    }
    race_fig = px.pie(
        race_data,
        values='Population',
        names='Race',
        title='Racial Distribution'
    )
    
    # Economics Chart
    economics_fig = go.Figure()
    economics_fig.add_trace(go.Bar(
        x=['Median Income', 'Per Capita Income', 'Median Home Value', 'Median Rent'],
        y=[
            selected_property['Median Household Income'],
            selected_property['Per Capita Income'],
            selected_property['Median Home Value'],
            selected_property['Median Gross Rent']
        ]
    ))
    economics_fig.update_layout(title='Economic Indicators')
    
    return (
        {'display': 'block'},
        description,
        precinct_info,
        subway_info,
        population_fig,
        race_fig,
        economics_fig
    )

if __name__ == '__main__':
    app.run_server(debug=True)