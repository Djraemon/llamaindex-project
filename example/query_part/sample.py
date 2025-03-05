
from llama_index.core.schema import TextNode

movie_nodes = [
    TextNode(
        text=(
            "A pragmatic paleontologist touring an almost complete theme park on an island "
            + "in Central America is tasked with protecting a couple of kids after a power "
            + "failure causes the park's cloned dinosaurs to run loose."
        ),
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    TextNode(
        text=(
            "A thief who steals corporate secrets through the use of dream-sharing technology "
            + "is given the inverse task of planting an idea into the mind of a C.E.O., "
            + "but his tragic past may doom the project and his team to disaster."
        ),
        metadata={
            "year": 2010,
            "director": "Christopher Nolan",
            "rating": 8.2,
        },
    ),
    TextNode(
        text="Barbie suffers a crisis that leads her to question her world and her existence.",
        metadata={
            "year": 2023,
            "director": "Greta Gerwig",
            "genre": "fantasy",
            "rating": 9.5,
        },
    ),
    TextNode(
        text=(
            "A cowboy doll is profoundly threatened and jealous when a new spaceman action "
            + "figure supplants him as top toy in a boy's bedroom."
        ),
        metadata={"year": 1995, "genre": "animated", "rating": 8.3},
    ),
    TextNode(
        text=(
            "When Woody is stolen by a toy collector, Buzz and his friends set out on a "
            + "rescue mission to save Woody before he becomes a museum toy property with his "
            + "roundup gang Jessie, Prospector, and Bullseye. "
        ),
        metadata={"year": 1999, "genre": "animated", "rating": 7.9},
    ),
    TextNode(
        text=(
            "The toys are mistakenly delivered to a day-care center instead of the attic "
            + "right before Andy leaves for college, and it's up to Woody to convince the "
            + "other toys that they weren't abandoned and to return home."
        ),
        metadata={"year": 2010, "genre": "animated", "rating": 8.3},
    ),
]

wiki_titles = ["Michael Jordan", "Elon Musk", "Richard Branson", "Rihanna"]
wiki_metadatas = {
    "Michael Jordan": {
        "category": "Sports",
        "country": "United States",
    },
    "Elon Musk": {
        "category": "Business",
        "country": "United States",
    },
    "Richard Branson": {
        "category": "Business",
        "country": "UK",
    },
    "Rihanna": {
        "category": "Music",
        "country": "Barbados",
    },
}