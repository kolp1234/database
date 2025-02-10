from sqlalchemy import ForeignKey, create_engine, Column, Integer, String, Float, Boolean, Date, BigInteger, Enum
from sqlalchemy.orm import relationship, sessionmaker
from dotenv import dotenv_values
from sqlalchemy.orm import declarative_base

username = st.secrets["DATABASE_USERNAME"]
password = st.secrets["DATABASE_PASSWORD"]
dbname = st.secrets["DATABASE_NAME"]
port = st.secrets["DATABASE_PORT"]
host = st.secrets["DATABASE_HOST"]

DATABASE_URL = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"

engine = create_engine(DATABASE_URL)

Base = declarative_base()

class Listing(Base):
    __tablename__ = 'listing'
    id = Column(BigInteger, primary_key=True)
    acres = Column(Float)
    price = Column(Float)
    sold_date = Column(Date, nullable=True)
    listing_date =  Column(Date, nullable=True)
    common_street = Column(String, nullable=True)
    street_address = Column(String, nullable=True)
    zipcode = Column(Float, nullable=True)
    city = Column(String, nullable=True)
    county = Column(String)
    state = Column(String)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    days_on_market = Column(Integer, nullable=True)
    site_name = Column(String, nullable=True)
    status = Column(String, nullable=True)
    url = Column(String, nullable=True)
    date_added_db = Column(Date) 

    
class Zillow_Coordinates(Base):
    __tablename__ = 'zillow_coordinates'
    id = Column(BigInteger, primary_key=True)
    county = Column(String)
    state = Column(String)
    west = Column(Float)
    east = Column(Float)
    south = Column(Float)
    north = Column(Float)
    region_id = Column(Integer)
    region_type = Column(Integer)

    
class Scrubbed(Base):
     __tablename__ = 'scrubbed'
     id = Column(BigInteger, primary_key=True)
     owner_first_name = Column(String)
     owner_last_name = Column(String, nullable=True)
     apn = Column(String, nullable=True)
     acreage = Column(Float)
     is_good = Column(Boolean, default= False, nullable=True)
     is_subdividable = Column(Boolean, default= False, nullable=True)
     is_commercial = Column(Boolean, default= False, nullable=True)
     wetland_issues = Column(Boolean, default= False, nullable=True)
     access_issues = Column(Boolean, default= False, nullable=True)
     bad_shape = Column(Boolean, default= False, nullable=True)
     neighbors_own = Column(Boolean, default= False, nullable=True)
     slope = Column(Boolean, default= False, nullable=True)
     house = Column(Boolean, default= False, nullable=True)
     transmission_line = Column(Boolean, default= False, nullable=True)
     no_info_found = Column(Boolean, default= False, nullable=True)
     misc_note = Column(String, nullable=True)
     latitude = Column(Float, nullable=True)
     longitude = Column(Float, nullable=True)
     county = Column(String)
     mail_zip = Column(String, nullable=True) 
     mail_addr = Column(String, nullable=True)
     mail_city = Column(String, nullable=True)
     mail_state = Column(String, nullable=True)
     prop_zip = Column(String, nullable=True)
     prop_addr = Column(String, nullable=True)
     prop_city = Column(String, nullable=True)
     prop_state = Column(String)
     land_sqft = Column(Float, nullable=True)
     val_transfer = Column(Integer, nullable=True)
     unit = Column(String, nullable=True)
     date_transfer = Column(Date, nullable=True)
     zoning = Column(String, nullable=True)
     owner_name_1 = Column(String, nullable=True)
     owner_name_2 = Column(String, nullable=True)
     val_assd_land = Column(Float, nullable=True)
     val_market_land = Column(Float, nullable=True)
     cal_acreage = Column(Float, nullable=True)
     use_code_muni = Column(Float, nullable=True)
     flood_factor = Column(Integer, nullable=True)
     imprv_pct = Column(Float, nullable=True)
     landuse_cat = Column(String, nullable=True)
     landuse_code = Column(Integer, nullable=True)
     landuse_desc = Column(String, nullable=True)
     legal_1 = Column(String, nullable=True)
     lot_number = Column(Integer, nullable=True)
     ownership_status_desc = Column(String, nullable=True)
     tax_amount = Column(BigInteger, nullable=True)
     tax_year = Column(BigInteger, nullable=True)
     simplified = Column(Boolean, default= False, nullable=True)
     ppa = Column(Float, nullable=True)
     offer_price =  Column(Float, nullable=True)
     est_market_value = Column(Float, nullable=True)
     county_state = Column(String, nullable=True)
     date_uploaded = Column(Date, nullable=False)
     date_modified = Column(Date, nullable=False)
     tags = Column(String, nullable=True)
     staging_status = Column(Enum("New", "Updated", "Unchanged", name="staging_status_enum"), nullable=True)
     demographic_data = relationship('Demographic_Append', back_populates='scrubbed')
     contact_data = relationship('Contact_Append', back_populates='scrubbed')
     

class Demographic_Append(Base):
    __tablename__ = 'demographic_append'
    id = Column(BigInteger, primary_key=True)
    date_of_birth = Column(Date, nullable=True)
    age = Column(Integer, nullable=True)
    age_range = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    ethnic_group = Column(String, nullable=True)
    religion = Column(String, nullable=True)
    education_level = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    language = Column(String, nullable=True)
    marital_status = Column(String, nullable=True)
    working_women_household = Column(Boolean, default= False, nullable=True)
    senior_woman_household = Column(Boolean, default= False, nullable=True)
    single_parent = Column(String, nullable=True)
    has_childfren =Column(Boolean, default= False, nullable=True)
    no_of_children = Column(Integer, nullable=True)
    young_adult_household = Column(Boolean, default=False, nullable=True)
    home_office = Column(Boolean, default= False, nullable=True)
    online_purchasing = Column(Boolean, default= False, nullable=True)
    online_education = Column(Boolean, default= False, nullable=True)
    scrubbed_id = Column(Integer, ForeignKey('scrubbed.id'))
    scrubbed = relationship('Scrubbed', back_populates='demographic_data')
    
class Contact_Append(Base):
     __tablename__ = 'contact_append'
     id = Column(BigInteger, primary_key=True)
     first_name = Column(String)
     last_name = Column(String)
     mob_phone_1 = Column(BigInteger)
     mob_phone_2 = Column(BigInteger)
     mob_phone_3 = Column(BigInteger)
     mob_phone_4 = Column(BigInteger)
     mob_phone_5 = Column(BigInteger)
     mob_phone_6 = Column(BigInteger)
     land_phone_1 = Column(BigInteger)
     land_phone_2 = Column(BigInteger)
     land_phone_3 = Column(BigInteger)
     land_phone_4 = Column(BigInteger)
     land_phone_5 = Column(BigInteger)
     land_phone_6 = Column(BigInteger)
     voip_phone_1 = Column(BigInteger)
     voip_phone_2 = Column(BigInteger)
     voip_phone_3 = Column(BigInteger)
     voip_phone_4 = Column(BigInteger)
     voip_phone_5 = Column(BigInteger)
     voip_phone_6 = Column(BigInteger)
     scrubbed_id = Column(BigInteger, ForeignKey('scrubbed.id'))
     scrubbed = relationship('Scrubbed', back_populates='contact_data')
    
    
# Create the table in the database
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)
session = Session()


