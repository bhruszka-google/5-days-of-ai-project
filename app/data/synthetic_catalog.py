"""Synthetic Dealership Inventory, Manufacturer Rebates, Credit Tiers, and Trade-In Matrix.

Provides a deterministic, realistic automotive dataset of 24 vehicles across
sedans, crossovers, 3-row family SUVs, electric vehicles, and pickup trucks
with strict Pydantic validation (`extra="forbid"`), OpenTelemetry tracing, and
Intent vs. Outcome structured logging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.observability.logger import log_intent, log_outcome
from app.observability.tracing import traced_operation


class VehicleRecord(BaseModel):
    """Strict schema representing a single vehicle on the dealership lot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vin: str = Field(
        ...,
        min_length=17,
        max_length=17,
        pattern=r"^[A-HJ-NPR-Z0-9]{17}$",
        description="17-character ISO standard Vehicle Identification Number.",
    )
    stock_number: str = Field(
        ...,
        min_length=5,
        description="Internal dealership stock identifier (e.g., STK-2026-001).",
    )
    year: int = Field(..., ge=2022, le=2027, description="Vehicle model year.")
    make: str = Field(..., min_length=2, description="Manufacturer brand name.")
    model: str = Field(..., min_length=1, description="Model name.")
    trim: str = Field(..., min_length=1, description="Trim package identifier.")
    body_style: Literal["sedan", "suv", "crossover", "truck", "ev"] = Field(
        ..., description="Primary vehicle body classification."
    )
    fuel_type: Literal["gas", "hybrid", "phev", "ev"] = Field(
        ..., description="Propulsion / fuel system type."
    )
    drivetrain: Literal["FWD", "AWD", "4WD", "RWD"] = Field(
        ..., description="Drivetrain layout."
    )
    condition: Literal["new", "cpo"] = Field(
        ..., description="Vehicle condition ('new' or certified pre-owned 'cpo')."
    )
    mileage: int = Field(..., ge=0, description="Current odometer reading in miles.")
    msrp: float = Field(
        ..., gt=0, description="Manufacturer Suggested Retail Price in USD."
    )
    invoice_price: float = Field(
        ..., gt=0, description="Dealer factory invoice cost in USD."
    )
    selling_price: float = Field(
        ..., gt=0, description="Current advertised dealership asking price in USD."
    )
    seating_capacity: int = Field(
        ..., ge=2, le=9, description="Total passenger seating capacity."
    )
    mpg_combined: int = Field(
        ..., ge=15, le=135, description="Combined city/highway MPG or MPGe rating."
    )
    safety_rating: float = Field(
        ..., ge=4.0, le=5.0, description="NHTSA 5-star overall safety rating."
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="Notable technology, comfort, and safety equipment tags.",
    )
    days_on_lot: int = Field(
        ..., ge=0, description="Number of days the unit has been in dealership stock."
    )

    @model_validator(mode="after")
    def validate_price_hierarchy(self) -> "VehicleRecord":
        """Enforces `msrp >= selling_price > invoice_price` across all inventory."""
        if self.selling_price > self.msrp:
            raise ValueError(
                f"selling_price ({self.selling_price}) cannot exceed msrp ({self.msrp})"
            )
        if self.invoice_price >= self.selling_price:
            raise ValueError(
                f"invoice_price ({self.invoice_price}) must be strictly less than "
                f"selling_price ({self.selling_price})"
            )
        return self

    @property
    def display_name(self) -> str:
        """Returns a formatted `Year Make Model Trim` title."""
        return f"{self.year} {self.make} {self.model} {self.trim}"


class ManufacturerRebate(BaseModel):
    """Strict schema representing an active manufacturer or dealer incentive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rebate_id: str = Field(..., min_length=3, description="Unique rebate code.")
    program_name: str = Field(..., min_length=3, description="Marketing program title.")
    amount: float = Field(..., gt=0, description="Dollar value of the rebate.")
    applicable_makes: List[str] = Field(
        default_factory=list,
        description="Eligible vehicle makes (or ['ALL'] for dealership-wide programs).",
    )
    applicable_fuel_types: List[str] = Field(
        default_factory=list,
        description="Eligible fuel types (or ['ALL'] if unrestricted).",
    )
    min_days_on_lot: int = Field(
        default=0, ge=0, description="Minimum lot age required to qualify."
    )
    customer_qualifier: Optional[str] = Field(
        default=None,
        description="Optional customer eligibility tag (e.g., 'loyalty', 'military', 'first_time_buyer').",
    )
    stackable: bool = Field(
        default=True,
        description="Whether this incentive can be combined with other programs.",
    )
    expiration_date: str = Field(
        default="2026-12-31", description="ISO date when the rebate expires."
    )


class CreditTierRate(BaseModel):
    """Strict schema representing APR and money-factor financing terms by credit tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier_id: Literal[
        "tier_1_excellent", "tier_2_good", "tier_3_fair", "tier_4_subprime"
    ]
    label: str
    min_credit_score: int = Field(..., ge=300, le=850)
    max_credit_score: int = Field(..., ge=300, le=850)
    apr_by_term_months: Dict[int, float] = Field(
        ..., description="Mapping of loan term (24, 36, 48, 60, 72) to annual APR %."
    )
    lease_money_factor: float = Field(
        ..., gt=0, description="36-month standard lease money factor."
    )
    lease_residual_pct_36m: float = Field(
        ...,
        gt=0,
        lt=1.0,
        description="Standard 36-month residual fraction of MSRP.",
    )


class TradeInReference(BaseModel):
    """Strict schema for baseline trade-in wholesale valuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    make: str
    model: str
    base_2022_value: float = Field(..., gt=0)
    annual_depreciation_rate: float = Field(..., gt=0, lt=0.35)
    popular_segment_bonus: float = Field(default=0.0, ge=0.0)


SYNTHETIC_VEHICLES: List[VehicleRecord] = [
    # --- 3-Row Family SUVs (Gas, Hybrid, PHEV, EV, CPO) ---
    VehicleRecord(
        vin="5TDKDRBH1TS100001",
        stock_number="STK-2026-001",
        year=2026,
        make="Toyota",
        model="Highlander Hybrid",
        trim="XLE",
        body_style="suv",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=14,
        msrp=48250.0,
        invoice_price=44390.0,
        selling_price=46800.0,
        seating_capacity=7,
        mpg_combined=35,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "captain chairs",
            "blind spot monitor",
            "power liftgate",
            "wireless carplay",
            "toyota safety sense 3.0",
        ],
        days_on_lot=42,
    ),
    VehicleRecord(
        vin="5TDAAAB52TS100002",
        stock_number="STK-2026-002",
        year=2026,
        make="Toyota",
        model="Grand Highlander Hybrid",
        trim="Limited",
        body_style="suv",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=18,
        msrp=53600.0,
        invoice_price=49310.0,
        selling_price=52150.0,
        seating_capacity=8,
        mpg_combined=34,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "panoramic roof",
            "ventilated leather seats",
            "jbl premium audio",
            "surround view camera",
            "hands-free liftgate",
        ],
        days_on_lot=28,
    ),
    VehicleRecord(
        vin="JM3KKEHD4TS100003",
        stock_number="STK-2026-003",
        year=2026,
        make="Mazda",
        model="CX-90 PHEV",
        trim="Preferred",
        body_style="suv",
        fuel_type="phev",
        drivetrain="AWD",
        condition="new",
        mileage=22,
        msrp=51400.0,
        invoice_price=47545.0,
        selling_price=49200.0,
        seating_capacity=7,
        mpg_combined=56,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "plug-in hybrid",
            "26-mile ev range",
            "leather interior",
            "head-up display",
            "adaptive cruise control",
        ],
        days_on_lot=68,
    ),
    VehicleRecord(
        vin="5XYP5DGC8TS100004",
        stock_number="STK-2026-004",
        year=2026,
        make="Kia",
        model="Sorento Hybrid",
        trim="EX",
        body_style="suv",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=11,
        msrp=42800.0,
        invoice_price=39590.0,
        selling_price=41250.0,
        seating_capacity=6,
        mpg_combined=37,
        safety_rating=4.8,
        key_features=[
            "3-row seating",
            "second-row captain chairs",
            "smart power tailgate",
            "highway driving assist",
            "heated seats",
            "remote start",
        ],
        days_on_lot=54,
    ),
    VehicleRecord(
        vin="5XYP6DGC0TS100005",
        stock_number="STK-2026-005",
        year=2026,
        make="Kia",
        model="Telluride",
        trim="EX X-Line",
        body_style="suv",
        fuel_type="gas",
        drivetrain="AWD",
        condition="new",
        mileage=19,
        msrp=47500.0,
        invoice_price=44175.0,
        selling_price=46100.0,
        seating_capacity=7,
        mpg_combined=22,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "tow package",
            "nappa leather",
            "dual sunroofs",
            "blind-spot view monitor",
            "driver talk intercom",
        ],
        days_on_lot=31,
    ),
    VehicleRecord(
        vin="5FNYG1H77SS100006",
        stock_number="STK-2025-006",
        year=2025,
        make="Honda",
        model="Pilot",
        trim="EX-L",
        body_style="suv",
        fuel_type="gas",
        drivetrain="AWD",
        condition="new",
        mileage=35,
        msrp=45900.0,
        invoice_price=42228.0,
        selling_price=43950.0,
        seating_capacity=8,
        mpg_combined=21,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "stowable middle seat",
            "leather seats",
            "wireless phone charger",
            "honda sensing suite",
            "acoustic glass",
        ],
        days_on_lot=79,
    ),
    VehicleRecord(
        vin="KM8R5DGE9TS100007",
        stock_number="STK-2026-007",
        year=2026,
        make="Hyundai",
        model="Santa Fe Hybrid",
        trim="SEL",
        body_style="suv",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=16,
        msrp=40950.0,
        invoice_price=37880.0,
        selling_price=39400.0,
        seating_capacity=7,
        mpg_combined=36,
        safety_rating=4.9,
        key_features=[
            "3-row seating",
            "dual wireless chargers",
            "digital key 2",
            "panoramic curved display",
            "blind spot warning",
            "roof rails",
        ],
        days_on_lot=47,
    ),
    VehicleRecord(
        vin="5TDKDRBH8RS100008",
        stock_number="STK-2024-008",
        year=2024,
        make="Toyota",
        model="Highlander Hybrid",
        trim="LE",
        body_style="suv",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="cpo",
        mileage=18400,
        msrp=43100.0,
        invoice_price=35200.0,
        selling_price=37900.0,
        seating_capacity=8,
        mpg_combined=35,
        safety_rating=5.0,
        key_features=[
            "3-row seating",
            "certified pre-owned warranty",
            "apple carplay",
            "blind spot monitor",
            "power liftgate",
            "clean 1-owner carfax",
        ],
        days_on_lot=64,
    ),
    # --- Compact & Midsize Crossovers (Hybrid & Gas) ---
    VehicleRecord(
        vin="2T3P1RFV4TS100009",
        stock_number="STK-2026-009",
        year=2026,
        make="Toyota",
        model="RAV4 Hybrid",
        trim="XSE",
        body_style="crossover",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=9,
        msrp=39600.0,
        invoice_price=36630.0,
        selling_price=38450.0,
        seating_capacity=5,
        mpg_combined=39,
        safety_rating=5.0,
        key_features=[
            "sport-tuned suspension",
            "two-tone roof",
            "softex heated seats",
            "10.5-inch touchscreen",
            "blind spot monitor",
            "all-weather mats",
        ],
        days_on_lot=21,
    ),
    VehicleRecord(
        vin="7FARS6H82TS100010",
        stock_number="STK-2026-010",
        year=2026,
        make="Honda",
        model="CR-V Hybrid",
        trim="Sport-L",
        body_style="crossover",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=12,
        msrp=39150.0,
        invoice_price=36210.0,
        selling_price=37800.0,
        seating_capacity=5,
        mpg_combined=40,
        safety_rating=5.0,
        key_features=[
            "leather-trimmed seats",
            "power tailgate",
            "parking sensors",
            "wireless carplay",
            "moonroof",
            "honda sensing",
        ],
        days_on_lot=38,
    ),
    VehicleRecord(
        vin="4S4BTANC1SS100011",
        stock_number="STK-2025-011",
        year=2025,
        make="Subaru",
        model="Outback",
        trim="Onyx Edition",
        body_style="crossover",
        fuel_type="gas",
        drivetrain="AWD",
        condition="new",
        mileage=27,
        msrp=36800.0,
        invoice_price=33850.0,
        selling_price=35100.0,
        seating_capacity=5,
        mpg_combined=29,
        safety_rating=5.0,
        key_features=[
            "symmetrical awd",
            "starTex water-repellent upholstery",
            "eyesight driver assist",
            "harman kardon audio",
            "heated steering wheel",
            "x-mode off-road",
        ],
        days_on_lot=63,
    ),
    VehicleRecord(
        vin="7MUAAAAG5TS100012",
        stock_number="STK-2026-012",
        year=2026,
        make="Mazda",
        model="CX-50 Hybrid",
        trim="Preferred",
        body_style="crossover",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=15,
        msrp=35900.0,
        invoice_price=33200.0,
        selling_price=34650.0,
        seating_capacity=5,
        mpg_combined=38,
        safety_rating=5.0,
        key_features=[
            "toyota hybrid powertrain",
            "e-awd",
            "heated front seats",
            "power rear liftgate",
            "alexa built-in",
            "radar cruise control",
        ],
        days_on_lot=29,
    ),
    VehicleRecord(
        vin="2T3R1RFV0RS100013",
        stock_number="STK-2024-013",
        year=2024,
        make="Toyota",
        model="RAV4",
        trim="XLE",
        body_style="crossover",
        fuel_type="gas",
        drivetrain="AWD",
        condition="cpo",
        mileage=15200,
        msrp=33400.0,
        invoice_price=27600.0,
        selling_price=29850.0,
        seating_capacity=5,
        mpg_combined=30,
        safety_rating=5.0,
        key_features=[
            "certified pre-owned",
            "convenience package",
            "blind spot monitor",
            "push button start",
            "multi-terrain select",
        ],
        days_on_lot=52,
    ),
    # --- Sedans (Compact & Midsize, Gas & Hybrid) ---
    VehicleRecord(
        vin="4T1DAACK9TS100014",
        stock_number="STK-2026-014",
        year=2026,
        make="Toyota",
        model="Camry",
        trim="SE Hybrid",
        body_style="sedan",
        fuel_type="hybrid",
        drivetrain="FWD",
        condition="new",
        mileage=8,
        msrp=32600.0,
        invoice_price=30150.0,
        selling_price=31450.0,
        seating_capacity=5,
        mpg_combined=50,
        safety_rating=5.0,
        key_features=[
            "5th-gen hybrid system",
            "sport mesh grille",
            "wireless apple carplay",
            "blind spot monitor",
            "18-inch black alloy wheels",
        ],
        days_on_lot=19,
    ),
    VehicleRecord(
        vin="1HGCY2F84TS100015",
        stock_number="STK-2026-015",
        year=2026,
        make="Honda",
        model="Accord Hybrid",
        trim="Touring",
        body_style="sedan",
        fuel_type="hybrid",
        drivetrain="FWD",
        condition="new",
        mileage=14,
        msrp=39900.0,
        invoice_price=36900.0,
        selling_price=38300.0,
        seating_capacity=5,
        mpg_combined=44,
        safety_rating=5.0,
        key_features=[
            "google built-in",
            "bose 12-speaker audio",
            "head-up display",
            "ventilated leather seats",
            "wireless phone charger",
        ],
        days_on_lot=45,
    ),
    VehicleRecord(
        vin="5YFB4MDE2SS100016",
        stock_number="STK-2025-016",
        year=2025,
        make="Toyota",
        model="Corolla",
        trim="LE",
        body_style="sedan",
        fuel_type="gas",
        drivetrain="FWD",
        condition="new",
        mileage=21,
        msrp=23950.0,
        invoice_price=22150.0,
        selling_price=22990.0,
        seating_capacity=5,
        mpg_combined=35,
        safety_rating=5.0,
        key_features=[
            "8-inch touchscreen",
            "wireless carplay",
            "automatic climate control",
            "led headlights",
            "toyota safety sense 3.0",
        ],
        days_on_lot=58,
    ),
    VehicleRecord(
        vin="KMHLN4AJ7SS100017",
        stock_number="STK-2025-017",
        year=2025,
        make="Hyundai",
        model="Elantra Hybrid",
        trim="Blue",
        body_style="sedan",
        fuel_type="hybrid",
        drivetrain="FWD",
        condition="new",
        mileage=17,
        msrp=27400.0,
        invoice_price=25480.0,
        selling_price=26350.0,
        seating_capacity=5,
        mpg_combined=54,
        safety_rating=4.9,
        key_features=[
            "54 mpg combined",
            "heated front seats",
            "hands-free smart trunk",
            "blind-spot collision warning",
            "dual-zone climate",
        ],
        days_on_lot=61,
    ),
    VehicleRecord(
        vin="2HGFE2F59RS100018",
        stock_number="STK-2024-018",
        year=2024,
        make="Honda",
        model="Civic",
        trim="EX",
        body_style="sedan",
        fuel_type="gas",
        drivetrain="FWD",
        condition="cpo",
        mileage=12800,
        msrp=27950.0,
        invoice_price=22900.0,
        selling_price=24600.0,
        seating_capacity=5,
        mpg_combined=36,
        safety_rating=5.0,
        key_features=[
            "certified pre-owned",
            "one-touch power moonroof",
            "heated front seats",
            "blind spot information system",
            "turbocharged engine",
        ],
        days_on_lot=34,
    ),
    # --- Battery Electric Vehicles (EVs) ---
    VehicleRecord(
        vin="KNDC4DLC6TS100019",
        stock_number="STK-2026-019",
        year=2026,
        make="Kia",
        model="EV9",
        trim="Land AWD",
        body_style="ev",
        fuel_type="ev",
        drivetrain="AWD",
        condition="new",
        mileage=15,
        msrp=69900.0,
        invoice_price=64650.0,
        selling_price=66800.0,
        seating_capacity=6,
        mpg_combined=89,
        safety_rating=5.0,
        key_features=[
            "3-row electric suv",
            "800v ultra-fast charging",
            "relaxation captain chairs",
            "meridian 14-speaker sound",
            "vehicle-to-load power",
        ],
        days_on_lot=74,
    ),
    VehicleRecord(
        vin="KM8KRDAF2TS100020",
        stock_number="STK-2026-020",
        year=2026,
        make="Hyundai",
        model="Ioniq 5",
        trim="SEL AWD",
        body_style="ev",
        fuel_type="ev",
        drivetrain="AWD",
        condition="new",
        mileage=10,
        msrp=52400.0,
        invoice_price=48470.0,
        selling_price=49950.0,
        seating_capacity=5,
        mpg_combined=99,
        safety_rating=5.0,
        key_features=[
            "303-mile range",
            "18-minute 10-80% fast charge",
            "nacs native port",
            "highway driving assist 2",
            "heated steering wheel",
        ],
        days_on_lot=66,
    ),
    VehicleRecord(
        vin="3GN7DLRP8TS100021",
        stock_number="STK-2026-021",
        year=2026,
        make="Chevrolet",
        model="Equinox EV",
        trim="LT",
        body_style="ev",
        fuel_type="ev",
        drivetrain="FWD",
        condition="new",
        mileage=12,
        msrp=35900.0,
        invoice_price=33560.0,
        selling_price=34750.0,
        seating_capacity=5,
        mpg_combined=108,
        safety_rating=4.9,
        key_features=[
            "319-mile EPA range",
            "17.7-inch diagonal display",
            "chevy safety assist",
            "regenerative one-pedal driving",
            "remote start",
        ],
        days_on_lot=49,
    ),
    # --- Pickup Trucks (Hybrid & Gas) ---
    VehicleRecord(
        vin="1FTFW1ED5TS100022",
        stock_number="STK-2026-022",
        year=2026,
        make="Ford",
        model="F-150",
        trim="XLT PowerBoost Hybrid",
        body_style="truck",
        fuel_type="hybrid",
        drivetrain="4WD",
        condition="new",
        mileage=20,
        msrp=58900.0,
        invoice_price=54180.0,
        selling_price=56200.0,
        seating_capacity=5,
        mpg_combined=24,
        safety_rating=4.9,
        key_features=[
            "pro power onboard 7.2kw",
            "max tow package",
            "360-degree camera",
            "ford co-pilot360",
            "heated cloth bench",
        ],
        days_on_lot=55,
    ),
    VehicleRecord(
        vin="3TMLB5JN1TS100023",
        stock_number="STK-2026-023",
        year=2026,
        make="Toyota",
        model="Tacoma",
        trim="TRD Off-Road",
        body_style="truck",
        fuel_type="gas",
        drivetrain="4WD",
        condition="new",
        mileage=11,
        msrp=44900.0,
        invoice_price=41530.0,
        selling_price=43600.0,
        seating_capacity=5,
        mpg_combined=21,
        safety_rating=4.8,
        key_features=[
            "bilstein off-road shocks",
            "locking rear differential",
            "crawl control",
            "composite bed with outlet",
            "14-inch touchscreen",
        ],
        days_on_lot=24,
    ),
    VehicleRecord(
        vin="3FTTW8E37SS100024",
        stock_number="STK-2025-024",
        year=2025,
        make="Ford",
        model="Maverick Hybrid",
        trim="XLT",
        body_style="truck",
        fuel_type="hybrid",
        drivetrain="AWD",
        condition="new",
        mileage=13,
        msrp=33200.0,
        invoice_price=30875.0,
        selling_price=32150.0,
        seating_capacity=5,
        mpg_combined=38,
        safety_rating=4.7,
        key_features=[
            "compact hybrid pickup",
            "flexbed storage system",
            "4k tow package",
            "apple carplay",
            "all-wheel drive hybrid",
        ],
        days_on_lot=37,
    ),
]


SYNTHETIC_REBATES: List[ManufacturerRebate] = [
    ManufacturerRebate(
        rebate_id="REB-CLEAN-EV-2026",
        program_name="Clean Vehicle EV & Plug-In Conquest Cash",
        amount=3500.0,
        applicable_makes=["ALL"],
        applicable_fuel_types=["ev", "phev"],
        min_days_on_lot=0,
        customer_qualifier=None,
        stackable=True,
    ),
    ManufacturerRebate(
        rebate_id="REB-HYBRID-BONUS-2026",
        program_name="Eco-Hybrid Retail Customer Cash",
        amount=1250.0,
        applicable_makes=["Toyota", "Honda", "Hyundai", "Kia", "Mazda", "Ford"],
        applicable_fuel_types=["hybrid"],
        min_days_on_lot=0,
        customer_qualifier=None,
        stackable=True,
    ),
    ManufacturerRebate(
        rebate_id="REB-AGED-LOT-60D",
        program_name="Aged Inventory 60+ Day Lot Clearance Bonus",
        amount=1000.0,
        applicable_makes=["ALL"],
        applicable_fuel_types=["ALL"],
        min_days_on_lot=60,
        customer_qualifier=None,
        stackable=True,
    ),
    ManufacturerRebate(
        rebate_id="REB-LOYALTY-TRADE",
        program_name="Owner Loyalty & Trade-In Assistance Bonus",
        amount=750.0,
        applicable_makes=["ALL"],
        applicable_fuel_types=["ALL"],
        min_days_on_lot=0,
        customer_qualifier="trade_in",
        stackable=True,
    ),
    ManufacturerRebate(
        rebate_id="REB-MILITARY-GRAD",
        program_name="Military & College Grad Appreciation Rebate",
        amount=500.0,
        applicable_makes=["ALL"],
        applicable_fuel_types=["ALL"],
        min_days_on_lot=0,
        customer_qualifier="military_or_grad",
        stackable=True,
    ),
]


CREDIT_TIER_RATES: Dict[str, CreditTierRate] = {
    "tier_1_excellent": CreditTierRate(
        tier_id="tier_1_excellent",
        label="Tier 1 Super Prime (720+)",
        min_credit_score=720,
        max_credit_score=850,
        apr_by_term_months={24: 2.9, 36: 3.49, 48: 3.99, 60: 4.49, 72: 4.99},
        lease_money_factor=0.00165,
        lease_residual_pct_36m=0.61,
    ),
    "tier_2_good": CreditTierRate(
        tier_id="tier_2_good",
        label="Tier 2 Prime (680-719)",
        min_credit_score=680,
        max_credit_score=719,
        apr_by_term_months={24: 4.25, 36: 4.75, 48: 5.25, 60: 5.79, 72: 6.29},
        lease_money_factor=0.00210,
        lease_residual_pct_36m=0.60,
    ),
    "tier_3_fair": CreditTierRate(
        tier_id="tier_3_fair",
        label="Tier 3 Near Prime (620-679)",
        min_credit_score=620,
        max_credit_score=679,
        apr_by_term_months={24: 6.50, 36: 6.99, 48: 7.49, 60: 7.99, 72: 8.49},
        lease_money_factor=0.00275,
        lease_residual_pct_36m=0.58,
    ),
    "tier_4_subprime": CreditTierRate(
        tier_id="tier_4_subprime",
        label="Tier 4 Subprime (<620)",
        min_credit_score=300,
        max_credit_score=619,
        apr_by_term_months={24: 9.99, 36: 10.49, 48: 11.25, 60: 11.99, 72: 12.75},
        lease_money_factor=0.00360,
        lease_residual_pct_36m=0.56,
    ),
}


TRADE_IN_BASE_VALUES: Dict[str, TradeInReference] = {
    "honda:cr-v": TradeInReference(
        make="Honda",
        model="CR-V",
        base_2022_value=24500.0,
        annual_depreciation_rate=0.075,
        popular_segment_bonus=800.0,
    ),
    "toyota:rav4": TradeInReference(
        make="Toyota",
        model="RAV4",
        base_2022_value=25200.0,
        annual_depreciation_rate=0.070,
        popular_segment_bonus=950.0,
    ),
    "toyota:camry": TradeInReference(
        make="Toyota",
        model="Camry",
        base_2022_value=21800.0,
        annual_depreciation_rate=0.080,
        popular_segment_bonus=500.0,
    ),
    "honda:civic": TradeInReference(
        make="Honda",
        model="Civic",
        base_2022_value=20400.0,
        annual_depreciation_rate=0.075,
        popular_segment_bonus=500.0,
    ),
    "ford:f-150": TradeInReference(
        make="Ford",
        model="F-150",
        base_2022_value=31500.0,
        annual_depreciation_rate=0.085,
        popular_segment_bonus=1000.0,
    ),
    "subaru:outback": TradeInReference(
        make="Subaru",
        model="Outback",
        base_2022_value=22600.0,
        annual_depreciation_rate=0.080,
        popular_segment_bonus=600.0,
    ),
    "toyota:highlander": TradeInReference(
        make="Toyota",
        model="Highlander",
        base_2022_value=30800.0,
        annual_depreciation_rate=0.075,
        popular_segment_bonus=900.0,
    ),
}


@traced_operation("autodeal.catalog.get_all_vehicles")
def get_all_vehicles() -> List[VehicleRecord]:
    """Returns a copy of all synthetic dealership vehicle records.

    Returns:
        List of `VehicleRecord` objects representing the dealership lot inventory.
    """
    return list(SYNTHETIC_VEHICLES)


@traced_operation("autodeal.catalog.get_vehicle_by_id_or_vin")
def get_vehicle_by_id_or_vin(identifier: str) -> Optional[VehicleRecord]:
    """Finds a vehicle by exact VIN, stock number, or case-insensitive display name.

    Args:
        identifier: VIN, stock number (`STK-...`), or vehicle name string.

    Returns:
        Matching `VehicleRecord` if found, otherwise `None`.
    """
    if not identifier or not identifier.strip():
        return None
    normalized = identifier.strip().upper()
    query_lower = identifier.strip().lower()

    for vehicle in SYNTHETIC_VEHICLES:
        if (
            vehicle.vin.upper() == normalized
            or vehicle.stock_number.upper() == normalized
        ):
            return vehicle

    for vehicle in SYNTHETIC_VEHICLES:
        full_title = vehicle.display_name.lower()
        if query_lower == full_title or query_lower in full_title:
            return vehicle

    return None


@traced_operation("autodeal.catalog.get_applicable_rebates")
def get_applicable_rebates(
    vehicle: VehicleRecord,
    customer_qualifiers: Optional[List[str]] = None,
) -> List[ManufacturerRebate]:
    """Determines all stackable manufacturer rebates applicable to a vehicle.

    Args:
        vehicle: Target `VehicleRecord` to evaluate.
        customer_qualifiers: Optional list of qualifier tags such as
            `["trade_in", "military_or_grad"]`.

    Returns:
        List of matching `ManufacturerRebate` records.
    """
    qualifiers_set = {q.strip().lower() for q in (customer_qualifiers or []) if q}
    log_intent(
        action_name="get_applicable_rebates",
        intended_arguments={
            "vin": vehicle.vin,
            "make": vehicle.make,
            "fuel_type": vehicle.fuel_type,
            "customer_qualifiers": sorted(qualifiers_set),
        },
        agent_name="inventory_specialist",
    )

    matched: List[ManufacturerRebate] = []
    for rebate in SYNTHETIC_REBATES:
        make_ok = (
            "ALL" in rebate.applicable_makes or vehicle.make in rebate.applicable_makes
        )
        fuel_ok = (
            "ALL" in rebate.applicable_fuel_types
            or vehicle.fuel_type in rebate.applicable_fuel_types
        )
        lot_age_ok = vehicle.days_on_lot >= rebate.min_days_on_lot
        qualifier_ok = (
            rebate.customer_qualifier is None
            or rebate.customer_qualifier.lower() in qualifiers_set
        )
        if make_ok and fuel_ok and lot_age_ok and qualifier_ok:
            matched.append(rebate)

    log_outcome(
        action_name="get_applicable_rebates",
        status="success",
        actual_outcome={
            "vin": vehicle.vin,
            "rebate_count": len(matched),
            "total_rebate_dollars": sum(r.amount for r in matched),
        },
        agent_name="inventory_specialist",
    )
    return matched


@traced_operation("autodeal.catalog.get_credit_tier_rate")
def get_credit_tier_rate(
    credit_tier: Optional[str] = None,
    credit_score: Optional[int] = None,
) -> CreditTierRate:
    """Resolves a customer's credit tier rate structure by tier name or FICO score.

    Args:
        credit_tier: Explicit credit tier key (e.g., `'tier_1_excellent'`).
        credit_score: Numeric FICO score (300-850) used if `credit_tier` is omitted.

    Returns:
        Matching `CreditTierRate` configuration.
    """
    if credit_tier:
        normalized = credit_tier.strip().lower()
        if normalized in CREDIT_TIER_RATES:
            return CREDIT_TIER_RATES[normalized]
        for key, tier in CREDIT_TIER_RATES.items():
            if normalized in key or normalized in tier.label.lower():
                return tier

    if credit_score is not None:
        if credit_score >= 720:
            return CREDIT_TIER_RATES["tier_1_excellent"]
        if credit_score >= 680:
            return CREDIT_TIER_RATES["tier_2_good"]
        if credit_score >= 620:
            return CREDIT_TIER_RATES["tier_3_fair"]
        return CREDIT_TIER_RATES["tier_4_subprime"]

    return CREDIT_TIER_RATES["tier_1_excellent"]


@traced_operation("autodeal.catalog.estimate_trade_in_value")
def estimate_trade_in_value(
    year: int,
    make: str,
    model: str,
    mileage: int,
    condition: str = "good",
) -> Dict[str, Any]:
    """Calculates a deterministic wholesale trade-in allowance for a customer vehicle.

    Args:
        year: Model year of the customer's trade-in vehicle (2010-2026).
        make: Vehicle manufacturer (e.g., `'Honda'`).
        model: Vehicle model name (e.g., `'CR-V'`).
        mileage: Current odometer reading in miles.
        condition: Cosmetic/mechanical condition (`'excellent'`, `'good'`, `'fair'`, `'poor'`).

    Returns:
        Dictionary containing `estimated_allowance`, `baseline_value`,
        `mileage_adjustment`, `condition_multiplier`, and `valuation_notes`.
    """
    lookup_key = f"{make.strip().lower()}:{model.strip().lower()}"
    ref = TRADE_IN_BASE_VALUES.get(
        lookup_key,
        TradeInReference(
            make=make.strip().title(),
            model=model.strip().title(),
            base_2022_value=21000.0,
            annual_depreciation_rate=0.085,
            popular_segment_bonus=300.0,
        ),
    )

    year_delta_from_2022 = year - 2022
    year_adjusted = ref.base_2022_value * (
        (1.0 + ref.annual_depreciation_rate * 0.85) ** year_delta_from_2022
        if year_delta_from_2022 >= 0
        else (1.0 - ref.annual_depreciation_rate) ** abs(year_delta_from_2022)
    )

    expected_mileage = max(1, (2026 - year)) * 12000
    mileage_diff = mileage - expected_mileage
    # Deduct $0.10 per excess mile or credit $0.06 per under-driven mile (capped)
    if mileage_diff > 0:
        mileage_adjustment = -min(6500.0, mileage_diff * 0.10)
    else:
        mileage_adjustment = min(2500.0, abs(mileage_diff) * 0.06)

    condition_multipliers = {
        "excellent": 1.06,
        "good": 1.00,
        "fair": 0.88,
        "poor": 0.74,
    }
    cond_key = condition.strip().lower()
    multiplier = condition_multipliers.get(cond_key, 1.00)

    raw_allowance = (
        year_adjusted + mileage_adjustment + ref.popular_segment_bonus
    ) * multiplier
    estimated_allowance = round(max(1500.0, raw_allowance), 2)

    return {
        "year": year,
        "make": ref.make,
        "model": ref.model,
        "mileage": mileage,
        "condition": cond_key,
        "baseline_value": round(year_adjusted, 2),
        "mileage_adjustment": round(mileage_adjustment, 2),
        "popular_segment_bonus": ref.popular_segment_bonus,
        "condition_multiplier": multiplier,
        "estimated_allowance": estimated_allowance,
    }
