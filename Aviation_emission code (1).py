import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px

@st.cache_data
def load_and_prepare_flights(max_distance_km=None):
    # -------------------------
    # Flights data (parquet)
    # -------------------------
    df5 = pd.read_parquet("flights_hackaton_20230701-20230801.parquet")
    df5 = df5[df5["ADEP"] == "EHAM"]

    df6 = pd.read_parquet("flights_hackaton_20230801-20230901.parquet")
    df6 = df6[df6["ADEP"] == "EHAM"]

    df_fl = pd.concat([df5, df6], ignore_index=True)

    # -------------------------
    # Sustainability data
    # -------------------------
    dftest = pd.read_csv(
        "LCC&FCC_Flights_Final_Sustainability_Score_and_Rating.csv"
    )
    dftest = dftest[dftest["ADEP"] == "EHAM"]

    # -------------------------
    # Merge
    # -------------------------
    dfm = df_fl[["ADES", "NAME_ADES"]]

    dfmerge = dftest.merge(dfm, on="ADES", how="left")

    # -------------------------
    # Kolommen herschikken
    # -------------------------
    dfmerge.insert(0, "Vertrek", "Schiphol")
    dfmerge.insert(1, "Bestemming", dfmerge.pop("NAME_ADES"))

    # -------------------------
    # Opschonen
    # -------------------------
    dfmerge = dfmerge.drop_duplicates()

    if max_distance_km is not None:
        dfmerge = dfmerge[
            dfmerge["Actual Distance Flown (km)"] <= max_distance_km
        ]

    dfmerge = dfmerge.drop(
        [
            "CO2 per FC seat (kg/km/seat)",
            "CO2 per PEC seat (kg/km/seat)",
            "Jet Engine type",
        ],
        axis=1,
        errors="ignore"
    )

    # -------------------------
    # Missing values invullen
    # -------------------------
    num_cols = dfmerge.select_dtypes(include="number").columns
    cat_cols = dfmerge.select_dtypes(exclude="number").columns

    dfmerge[num_cols] = dfmerge.groupby("Bestemming")[num_cols].transform(
        lambda x: x.fillna(x.median())
    )

    dfmerge[cat_cols] = dfmerge.groupby("Bestemming")[cat_cols].transform(
        lambda x: x.fillna(
            x.mode().iloc[0] if not x.mode().empty else np.nan
        )
    )
    rating_map = {
    "A": 7,
    "B": 6,
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "G": 1
    }

    dfmerge['CO2 Rating Num'] = dfmerge["CO2 Rating"].map(rating_map)
    dfmerge['Sustainability rating num'] = dfmerge['Sustainability Rating'].map(rating_map)
    airline_map = {
    "KLM": "KLM Royal Dutch Airlines",
    "TRA": "Transavia",
    "SXS": "SunExpress",
    "AEE": "Aegean Airlines",
    "VLG": "Vueling Airlines",
    "EZS": "easyJet Switzerland",
    "ASL": "ASL Airlines",
    "AMC": "Air Malta Charter",
    "ROT": "TAROM",
    "RYR": "Ryanair",
    "EIN": "Aer Lingus",
    "DLH": "Lufthansa",
    "CPA": "Cathay Pacific",
    "FIN": "Finnair",
    "THY": "Turkish Airlines",
    "PGT": "Pegasus Airlines",
    "ICE": "Icelandair",
    "TAP": "TAP Air Portugal",
    "BAW": "British Airways",
    "AEA": "Air Europa",
    "IBE": "Iberia",
    "AFR": "Air France",
    "UAE": "Emirates",
    "BTI": "airBaltic",
    "AUA": "Austrian Airlines",
    "LOT": "LOT Polish Airlines",
    "CTN": "Croatia Airlines",
    "SWR": "SWISS International Air Lines"
    }
    dfmerge["AC Operator"] = dfmerge["AC Operator"].map(airline_map)
    dfmerge["Route"] = dfmerge["Vertrek"] + " → " + dfmerge["Bestemming"]
    dfmerge['CO2 per passenger'] = dfmerge['CO2 per Passenger (kg/km/passenger)'] * dfmerge['Actual Distance Flown (km)']
    return dfmerge.sort_values("Bestemming").reset_index(drop=True)

@st.cache_data
def treinroutes():
    treinroutes = {

    "Billund": [
        {"van": "Amsterdam", "naar": "Osnabrück", "vervoer": "an ICE", "km": 215},
        {"van": "Osnabrück", "naar": "Hamburg", "vervoer": "an ICE", "km": 192},
        {"van": "Hamburg", "naar": "Flensburg", "vervoer": "an Intercity", "km": 141},
        {"van": "Flensburg", "naar": "Kolding", "vervoer": "an Intercity", "km": 80},
        {"van": "Kolding", "naar": "Vejle", "vervoer": "an Intercity", "km": 45},
        {"van": "Vejle", "naar": "Billund", "vervoer": "a Bus", "km": 33},
    ],
    "Birmingham": [
        {"van": "Amsterdam", "naar": "Brussels", "vervoer": "a Eurostar", "km": 176},
        {"van": "Brussels", "naar": "London", "vervoer": "a Eurostar", "km": 317},
        {"van": "London", "naar": "Birmingham", "vervoer": "a Regional train", "km": 163}
    ],

    "Bremen": [
        {"van": "Amsterdam", "naar": "Osnabrück", "vervoer": "an ICE", "km": 215},
        {"van": "Osnabrück", "naar": "Bremen", "vervoer": "an ICE", "km": 103}
    ],

    "Brussels": [
        {"van": "Amsterdam", "naar": "Brussels", "vervoer": "a Eurostar", "km": 176}
    ],

    "Düsseldorf": [
        {"van": "Amsterdam", "naar": "Arnhem", "vervoer": "an Intercity", "km": 97},
        {"van": "Arnhem", "naar": "Düsseldorf", "vervoer": "an Intercity", "km": 105}
    ],

    "Frankfurt": [
        {"van": "Amsterdam", "naar": "Cologne", "vervoer": "an ICE", "km": 214},
        {"van": "Cologne", "naar": "Frankfurt", "vervoer": "an ICE", "km": 152}
    ],


    "Hamburg": [
        {"van": "Amsterdam", "naar": "Osnabrück", "vervoer": "an ICE", "km": 215},
        {"van": "Osnabrück", "naar": "Hamburg", "vervoer": "an ICE", "km": 192}
    ],

    "Hanover": [
        {"van": "Amsterdam", "naar": "Hannover", "vervoer": "an ICE", "km": 329}
    ],    

    "London": [
        {"van": "Amsterdam", "naar": "Brussels", "vervoer": "a Eurostar", "km": 176},
        {"van": "Brussels", "naar": "London", "vervoer": "a Eurostar", "km": 317}
    ],


    "Luxembourg": [
        {"van": "Amsteram", "naar": "Brussels", "vervoer": "a Eurostar", "km": 176},
        {"van": "Brussels", "naar": "Luxembourg", "vervoer": "an Intercity", "km": 186}   
    ],


    "Paris": [
        {"van": "Amsterdam", "naar": "Paris", "vervoer": "a Eurostar", "km": 431},
        
    ],
    }

    treinuitstoot = {
        "a Eurostar": 0.009,
        "an Intercity": 0.0114,
        "a Regional train": 0.0292,
        "a Bus": 0.0947,
        "an ICE": 0.0032
    }


    resultaten = []

    for bestemming, segments in treinroutes.items():
        totale_co2 = 0

        for seg in segments:
            km = seg["km"]
            vervoer = seg["vervoer"]
            totale_co2 += km * treinuitstoot[vervoer]

        resultaten.append({
            "Bestemming": bestemming,
            "Totale_CO2_kg": round(totale_co2, 2)
        })

    df_trein_co2 = pd.DataFrame(resultaten)

    return treinroutes, treinuitstoot, df_trein_co2

# Met 500 km limiet
if "dfmerge_500" not in st.session_state:
    st.session_state.dfmerge_500 = load_and_prepare_flights(
        max_distance_km=500
    )

# Zonder afstandslimiet
if "dfmerge_all" not in st.session_state:
    st.session_state.dfmerge_all = load_and_prepare_flights(
        max_distance_km=None
    )

if "dftrain" not in st.session_state:
    _,_,st.session_state.dftrain = treinroutes()

if "route" not in st.session_state:
    st.session_state.route,_,_ = treinroutes()

if "uitstoot" not in st.session_state:
    _,st.session_state.uitstoot,_ = treinroutes()




df_500 = st.session_state.dfmerge_500
df_all = st.session_state.dfmerge_all
treinroutes = st.session_state.route
treinuitstoot = st.session_state.uitstoot
dfuitstoot = st.session_state.dftrain

st.set_page_config(layout="wide")

st.title("Aviation Emission Dashboard")

st.sidebar.title("Introduction 📖")
st.sidebar.markdown("""
Sustainability has become a key priority in modern mobility planning.  
The aviation sector is responsible for approximately **3.5% of global CO₂ emissions**, driving the need for more sustainable transport solutions.

This dashboard explores how **short-haul flights from Schiphol Amsterdam** could be replaced by **lower-emission train connections** within Europe.  
It provides data-driven insights to support **governments and policy makers** in reducing aviation-related emissions.

### What this dashboard shows
- ✈️ Short haul flights departing from **Schiphol Amsterdam**
- 📊 Ranking of **engines, aircraft and airlines**
- 🌱 Environmental performance based on:
  1. CO₂ rating  
  2. CO₂ emissions per passenger  
  3. Sustainability rating
- 🚆 Replacement of all flights under **500 km** with train routes
- ⚖️ Comparison of **CO₂ emissions per passenger** between air and rail
  """)


st.header("✈️ Flying routes")
st.write("Note: All flights are departing from Schiphol and are under 500 km")

dfmap = df_500.copy()
dfmap = dfmap.drop_duplicates(subset = ['Vertrek', 'Bestemming'])
dfmap = dfmap[['Vertrek', 'Bestemming', 'ADEP Latitude', 'ADEP Longitude', 'ADES Latitude', 'ADES Longitude', "Route"]]

df = dfmap.copy()

routes = df["Route"].unique()
gekozen_route = st.selectbox(
    "Select a flight",
    ["No selection"] + sorted(list(routes))
)

# -------------------------
# FOLIUM MAP
# -------------------------
m = folium.Map(
    location=[52.3676, 5.2],  # Amsterdam
    zoom_start=5,
    tiles="cartodbdark_matter"
)

# -------------------------
# ROUTES TEKENEN
# -------------------------
for _, row in df.iterrows():
    vertrek = [row["ADEP Latitude"], row["ADEP Longitude"]]
    bestemming = [row["ADES Latitude"], row["ADES Longitude"]]

    if row["Route"] == gekozen_route:
        kleur = "lime"
        gewicht = 3
        opacity = 0.9
        zindex = 10
    else:
        kleur = "gray"
        gewicht = 2
        opacity = 0.4
        zindex = 1

    folium.PolyLine(
        locations=[vertrek, bestemming],
        color=kleur,
        weight=gewicht,
        opacity=opacity,
        tooltip=row["Route"],
        z_index=zindex
    ).add_to(m)

# -------------------------
# MAP TONEN
# -------------------------
st_folium(m, width=1500, height=700)

dfroute = df_500.copy()
if gekozen_route != "No selection":
    # Filter op de gekozen route
    df_route = dfroute[dfroute["Route"] == gekozen_route].copy()
    
 
    df_route = df_route.sort_values(
        by=['CO2 Rating Num', 'CO2 per passenger','Sustainability rating num'],
        ascending=[False, True, True] 
    )
    df_route = df_route.drop_duplicates(subset = ['CO2 Rating Num', 'CO2 per passenger','AC Operator','Aircraft Variant','Engine Model'])
    df_route['Actual Distance Flown (km)'] = df_route['Actual Distance Flown (km)'].round(2)
    df_route['CO2 per passenger'] = df_route['CO2 per passenger'].round(2)

    rows = len(df_route)

    row_height = 35          # px per rij (werkt goed)
    header_height = 40       # px voor kolomheaders
    max_height = 300         # maximale hoogte

    height = min(
        header_height + rows * row_height,
        max_height
    )
    st.subheader(f"Flights ordered on environmental impact with route: {gekozen_route}")
    st.dataframe(df_route[['AC Operator','AC Type','Aircraft Variant','Engine Model','Actual Distance Flown (km)', 'CO2 Rating','CO2 per passenger', 'Sustainability Rating' ]].reset_index(drop=True), height=height)
else:
    st.info("Select a flight to view the corresponding data.")
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("How do the labels/ratings work?")
    st.markdown("""
    The A-G environmental label gives insight into the impact flights have on the environment,  
    giving a useful insight to airlines, airports, and governments.

    Flights labeled **A** are the most eco-friendly, having the lowest CO₂ emissions per passenger.  
    Flights with a **G** rating have the worst impact on the environment.

    The rating is based on various factors such as fuel efficiency, route length, passenger load, and aircraft type.  
    These ratings provide insights to governments, airlines, and airports on the impact these flights have,  
    allowing them to make decisions to achieve environmental goals and greener aviation strategies.
    """)
with col2:

    image = Image.open("64367e1e0c618f5102158c8c_Saman_EnergiQ_Energielabels_Tekengebied-1-1600x1080.jpeg")

    # Toon afbeelding in Streamlit
    st.image(image,  width = 300)

tab1, tab2, tab3, tab4 = st.tabs(['✈️ Flight replacements','🛠️ Engines','🛫 Airlines','🛩️ Aircrafts'])
with tab1:
    st.header("Replacing flights with trains")
    st.write("Note: These only contain flights departing from Schiphol under 500 km.")
    st.subheader("Choose the route you would like to change from plane to train:")


    bestemming = st.selectbox(
        "Choose a destination:",
        ["No selection"] + sorted(treinroutes.keys())
    )
    if bestemming != "No selection":

        st.subheader(f"Trainroute to {bestemming}")

        totale_km = 0
        totale_co2 = 0
        
        for i, segment in enumerate(treinroutes[bestemming], start=1):
            km = segment["km"]
            vervoer = segment["vervoer"]
            co2 = km * treinuitstoot[vervoer]

            totale_km += km
            totale_co2 += co2

            st.markdown(
                f"**Step {i}:** From **{segment['van']}** to **{segment['naar']}** "
                f"with **{vervoer}** ({km} km) "
                f"→ **{co2:.2f} kg CO₂**"
            )


        st.markdown("### Total journey")
        st.markdown(f"- **Total distance:** {totale_km} km")
        st.markdown(f"- **Total CO₂-emission:** *{totale_co2:.2f} kg CO₂*")
    else:
        st.info("Select a flight to view the corresponding train replacement.")

    st.markdown("---")

    flight = df_500.copy()
    flightco2 = (
        flight
        .groupby("Bestemming")["Total CO2 Emissions (kg)"]
        .mean()
        ).reset_index()

    flight = df_500.copy()
    flightco2 = (
    flight
    .groupby("Bestemming")["CO2 per passenger"]
    .mean()
    ).reset_index()


    st.subheader("Comparison CO2 emission of flights en train journeys")
    st.write("Note: These only contain flights departing fro Schiphol under 500 km")
    col1, col2 = st.columns(2)

    with col1:

        
        fig_train = px.bar(
            dfuitstoot,
            x="Totale_CO2_kg",
            y="Bestemming",
            title="Train CO₂ emissions per passenger of each destination",
            labels={
                    "Bestemming": "Destination",
                    "Totale_CO2_kg": "CO₂ emissions per passenger (kg/pass)"
                },
            color_discrete_sequence=["green"],
        )

        fig_train.update_layout(
            template="plotly_white",
            xaxis_tickangle=-45
            )

        st.plotly_chart(fig_train)

    with col2:


        fig_plane = px.bar(
            flightco2,
            x="CO2 per passenger",
            y="Bestemming",
            title="Flight CO₂ emissions per passenger of each destination",
            labels={
                "Bestemming": "Destination",
                "Total CO2 Emissions (kg)": "CO₂ emissions per passenger (kg/pass)"
            },
            color_discrete_sequence=["red"]
        )

        fig_plane.update_layout(
                xaxis_title = "CO₂ emissions per passenger (kg/pass)",
                template="plotly_white",
                xaxis_tickangle=-45
            )

        st.plotly_chart(fig_plane, use_container_width=True)


with tab2:
    st.header("Engine Insights")
    st.write("Note: This only contains flights departing from Schiphol.")
    st.subheader("Which plane engines are the best for the environment?")


    with st.expander("🌱 Click here to see the 3 best engine models for the environment"):
        st.markdown("""
    **The best engine models for the environment are:**

    1. **PW1127G**
    2. **V2527-A5**
    3. **CFM56-LEAP-1A32**

    This top 3 is determined by an **overall score** based on the **three main environmental metrics**.
    """)

    st.markdown("---")
    dfeng = df_all.copy()
    dfeng = (dfeng.groupby("Engine Model")[['CO2 Rating Num', 'CO2 per passenger', 'Sustainability rating num']].mean()).reset_index()
    dfeng = dfeng[dfeng['Engine Model'] != "Unknown"]

    dfeng = dfeng.sort_values('CO2 Rating Num', ascending = False)

    fig_rating = px.bar(
        dfeng,
        x="Engine Model",
        y="CO2 Rating Num",
        color="CO2 Rating Num",
        color_continuous_scale="RdYlGn",
        labels={
            "Engine Model":"Engine Model",
            "CO2 Rating Num":"CO₂ Rating"
        }
    )



    fig_rating.update_layout(
        title = "Average CO2 rating of each Engine model",
        yaxis_title = "CO2 Rating (higher = better)",
        legend_title_text = "CO2 Rating",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_rating)

    dfeng = dfeng.sort_values('CO2 per passenger', ascending = True)
    fig_flow = px.bar(
        dfeng,
        x="Engine Model",
        y='CO2 per passenger',
        color='CO2 per passenger',
        color_continuous_scale="RdYlGn_r",
        labels={
            "Engine Model":"Engine Model",
            "CO2 per passenger":"CO2 emission"
        }
    )



    fig_flow.update_layout(
        title = "Average CO2 emission per passenger in take off of each engine model",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_flow)

    dfeng = dfeng.sort_values('Sustainability rating num', ascending = False)
    fig_sus = px.bar(
        dfeng,
        x="Engine Model",
        y='Sustainability rating num',
        color='Sustainability rating num',
        color_continuous_scale="RdYlGn",
        labels={
            "Engine Model":"Engine Model",
            'Sustainability rating num':"Sustainability rating"
        }
    )



    fig_sus.update_layout(
        title = "Average sustainability rating of each engine model",
        yaxis_title = "Sustainability rating (higher = better)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_sus)


# Laatste 2 Tabs!!
with tab3:
    st.header("Airliner insights")
    st.write("Note: This only contains flights departing from Schiphol.")
    st.subheader("Which Aircraft operator are the best for the environment?")

    with st.expander("🌱 Click here to see the 3 best airlines for the environment"):
        st.markdown("""
    **The best airlines for the environment are:**

    1. **Vueling Airlines**
    2. **Pegasus Airlines**
    3. **easyjet Switzerland**

    This top 3 is determined by an **overall score** based on the **three main environmental metrics**.
    """)
       
    st.markdown("---") 

    dfairl = df_all.copy()
    dfairl = dfairl.dropna(subset = ['AC Operator'])
    dfairlsc = (dfairl.groupby('AC Operator')[['CO2 Rating Num', 'CO2 per passenger','Sustainability rating num' ]].mean()).reset_index()

    dfairlc = dfairlsc.sort_values('CO2 Rating Num', ascending = False)
    fig_rating = px.bar(
        dfairlc,
        x="AC Operator",
        y='CO2 Rating Num',
        color='CO2 Rating Num',
        color_continuous_scale="RdYlGn",
        labels={
            "AC Operator":"Aircraft operator",
            "CO2 Rating Num":"CO₂ Rating"
        }
    )
    fig_rating.update_layout(
        title = "Average CO2 rating of each aircraft operator",
        yaxis_title = "CO2 Rating (hoger = beter)",
        legend_title_text = "CO2 Rating",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_rating)

    dfairlc = dfairlsc.sort_values('CO2 per passenger', ascending = True)
    fig_rating = px.bar(
        dfairlc,
        x="AC Operator",
        y='CO2 per passenger',
        color='CO2 per passenger',
        color_continuous_scale="RdYlGn_r",
        labels={
            "AC Operator":"Aircraft operator",
            "CO2 per passenger":"CO₂ emission (kg/pass)"
        }
    )
    fig_rating.update_layout(
        title = "Average CO2 emission per passenger of each aircraft operator",
        yaxis_title = "CO2 emission (kg/pass)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_rating)

    dfairlc = dfairlsc.sort_values('Sustainability rating num', ascending = False)
    fig_sus = px.bar(
        dfairlc,
        x="AC Operator",
        y='Sustainability rating num',
        color='Sustainability rating num',
        color_continuous_scale="RdYlGn",
        labels={
            "AC Operator":"Aircraft operator",
            'Sustainability rating num':"Sustainability rating"
        }
    )

    st.markdown("---") 

    fig_sus.update_layout(
        title = "Average sustainability rating of each aircraft operator",
        yaxis_title = "Sustainability rating (higher = better)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_sus)

with tab4:
    st.header("Aircraft insights")
    st.write("Note: This only contains flights departing from Schiphol.")
    st.subheader("Which Aircrafts are the best for the environment?")

    with st.expander("🌱 Click here to see the 3 best aircraft variants for the environment"):
        st.markdown("""
    **The best engines for the environment are:**

    1. **A320neo**
    2. **A220-300**
    3. **A321neo ACF**

    This top 3 is determined by an **overall score** based on the **three main environmental metrics**.
    """)
        
    
    dffull = df_all.copy()
    dfvar = (
    dffull
    .groupby("Aircraft Variant")[['CO2 Rating Num','CO2 per passenger', 'Sustainability rating num' ]].mean()
    ).reset_index()
    
    dfvar = dfvar[dfvar['Aircraft Variant'] != "Unknown"]

    dfvar = dfvar.sort_values('CO2 Rating Num', ascending = False)

    fig_rating = px.bar(
        dfvar,
        x="Aircraft Variant",
        y="CO2 Rating Num",
        color="CO2 Rating Num",
        color_continuous_scale="RdYlGn",
        labels={
            "Aircraft Viarant":"Aircraft Variant",
            "CO2 Rating Num":"CO₂ Rating"
        }
    )



    fig_rating.update_layout(
        title = " Average CO2 rating of each aircraft variant",
        yaxis_title = "CO2 rating (hoger is beter)",
        legend_title_text = "CO2 Rating",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    st.plotly_chart(fig_rating)

    dfvar = dfvar.sort_values("CO2 per passenger", ascending = True)
    fig_passenger = px.bar(
        dfvar,
        x="Aircraft Variant",
        y="CO2 per passenger",
        title="Average CO₂ per passenger of each Aircraft vairant",
        labels={
            "AC Type": "Aircraft Variant",
            "CO2 per passenger": "CO₂ Emission"
        },
        color="CO2 per passenger",               
        color_continuous_scale="RdYlGn_r"
    )

    fig_passenger.update_layout(
        yaxis_title = "CO2 Emission per passenger (kg/pass)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_passenger)


    
    dfvar = dfvar.sort_values("Sustainability rating num", ascending = False)
    fig_passenger = px.bar(
        dfvar,
        x="Aircraft Variant",
        y="Sustainability rating num",
        title="Average sustainability rating of each Aircraft variant",
        labels={
            "AC Type": "Aircraft Variant",
            "Sustainability rating num": "Sustainability rating"
        },
        color="Sustainability rating num",               
        color_continuous_scale="RdYlGn"
    )

    fig_passenger.update_layout(
        yaxis_title = "Sustainability rating (higher = better)",
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig_passenger)