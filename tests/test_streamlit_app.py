def test_streamlit_app_import():
    """
    Test simple : vérifier que le module Streamlit s'importe correctement
    et que l'URL par défaut de l'API est définie.
    """
    import app.app as streamlit_app   

    assert hasattr(streamlit_app, "API_URL_DEFAULT")
    assert isinstance(streamlit_app.API_URL_DEFAULT, str)
    assert "http" in streamlit_app.API_URL_DEFAULT
