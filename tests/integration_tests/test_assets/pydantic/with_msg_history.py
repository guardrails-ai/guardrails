from pydantic import BaseModel, Field


class Movie(BaseModel):
    # """Details about a movie."""
    name: str = Field(..., description="The name of the movie.")
    director: str = Field(..., description="The name of the director.")
    release_year: int = Field(..., description="The year the movie was released.")
