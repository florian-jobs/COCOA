from src.DataAugmentation import get_cleaned_text, create_index

def test_get_cleaned_text_strips_stopwords():
    cleaned = get_cleaned_text("The quick brown fox is on the table.")
    assert cleaned == "The quick brown fox table"

def test_create_index_returns_one_entry_per_value():
    min_index, order_list, binary_list = create_index(["3", "1", "2", "2"])
    assert len(order_list) == 4
    assert len(binary_list) == 4
