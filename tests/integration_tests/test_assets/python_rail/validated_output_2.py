import datetime

llm_2_out = {
    "name": "Christopher Nolan",
    "movies": [
        {
            "rank": 1,
            "title": "Inception",
            "details": {
                "release_date": datetime.date(2010, 7, 16),
                "duration": datetime.time(2, 28),
                "budget": 160000000.0,
                "is_sequel": False,
                "website": "https://www.inceptionmovie.com",
                "contact_email": "info@inceptionmovie.com",
                "revenue_type": "box_office",
                "box_office": {
                    "gross": 829895144.0,
                    "opening_weekend": 62785337.0
                }
            }
        },
        {
            "rank": 2,
            "title": "The Dark Knight",
            "details": {
                "release_date": datetime.date(2008, 7, 18),
                "duration": datetime.time(2, 32),
                "budget": 185000000.0,
                "is_sequel": True,
                "website": "https://www.thedarkknightmovie.com",
                "contact_email": "info@thedarkknightmovie.com",
                "revenue_type": "box_office",
                "box_office": {
                    "gross": 1004558444.0,
                    "opening_weekend": 158411483.0
                }
            }
        },
        {
            "rank": 3,
            "title": "The Dark Knight Rises",
            "details": {
                "release_date": datetime.date(2012, 7, 20),
                "duration": datetime.time(2, 44),
                "budget": 250000000.0,
                "is_sequel": True,
                "website": "https://www.thedarkknightrises.com",
                "contact_email": "info@thedarkknightrises.com",
                "revenue_type": "streaming",
                "streaming": {
                    "subscriptions": 15000000,
                    "subscription_fee": 9.99
                }
            }
        },
        {
            "rank": 4,
            "title": "Interstellar",
            "details": {
                "release_date": datetime.date(2014, 11, 7),
                "duration": datetime.time(2, 49),
                "budget": 165000000.0,
                "is_sequel": False,
                "website": "https://www.interstellarmovie.com",
                "contact_email": "info@interstellarmovie.com",
                "revenue_type": "box_office",
                "box_office": {
                    "gross": 115000000.0,
                    "opening_weekend": 47510360.0
                }
            }
        },
        {
            "rank": 5,
            "title": "Dunkirk",
            "details": {
                "release_date": datetime.date(2017, 7, 21),
                "duration": datetime.time(1, 46),
                "budget": 100000000.0,
                "is_sequel": False,
                "website": "https://www.dunkirkmovie.com",
                "contact_email": "info@dunkirkmovie.com",
                "revenue_type": "box_office",
                "box_office": {
                    "gross": 526940665.0,
                    "opening_weekend": 50513488.0
                }
            }
        }
    ]
}