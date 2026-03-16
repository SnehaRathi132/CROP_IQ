(function () {
  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(el, message) {
    if (!el) return;
    el.textContent = message;
  }

  function getLang() {
    var stored = localStorage.getItem("lang");
    return stored === "hi" ? "hi" : "en";
  }

  var MESSAGES = {
    geolocationUnsupported: {
      en: "Geolocation is not supported in this browser.",
      hi: "इस ब्राउज़र में लोकेशन सपोर्ट नहीं है।",
    },
    fetching: {
      en: "Fetching your location, weather, and soil data...",
      hi: "आपकी लोकेशन, मौसम और मिट्टी की जानकारी ली जा रही है...",
    },
    weatherUpdated: {
      en: "Weather updated from your location.",
      hi: "लोकेशन से मौसम अपडेट हुआ।",
    },
    weatherFailed: {
      en: "Unable to fetch weather. You can enter it manually.",
      hi: "मौसम नहीं मिल पाया। आप मान स्वयं भर सकते हैं।",
    },
    soilUpdated: {
      en: "Weather and soil conditions auto-filled.",
      hi: "मौसम और मिट्टी की जानकारी स्वतः भर गई।",
    },
    soilFailed: {
      en: "Soil data unavailable. We'll estimate values from soil type.",
      hi: "मिट्टी की जानकारी नहीं मिली। हम मिट्टी के प्रकार से अनुमान लगाएंगे।",
    },
    cityUpdated: {
      en: "City auto-filled from your location.",
      hi: "लोकेशन से शहर स्वतः भर गया।",
    },
    locationDenied: {
      en: "Unable to access location. You can enter weather manually.",
      hi: "लोकेशन नहीं मिल सकी। आप मौसम मान स्वयं भर सकते हैं।",
    },
  };

  function msg(key) {
    var lang = getLang();
    return MESSAGES[key] ? MESSAGES[key][lang] : "";
  }

  function bindWeather(prefix) {
    var button = byId(prefix + "-locate");
    if (!button) return;

    var status = byId(prefix + "-weather-status");
    var tempInput = byId(prefix + "-temperature");
    var humidityInput = byId(prefix + "-humidity");
    var nitrogenInput = byId("Nitrogen");
    var phosphorousInput = byId("Phosphorous");
    var pottasiumInput = byId("Pottasium");
    var phInput = byId("ph");
    var cityInput = byId("City");
    var soilTypeSelect = byId("SoilType");
    var latInput = byId(prefix + "-lat");
    var lonInput = byId(prefix + "-lon");

    button.addEventListener("click", function () {
      console.log("Use My Location button clicked");
      
      if (!navigator.geolocation) {
        console.error("Geolocation not supported");
        setStatus(status, msg("geolocationUnsupported"));
        return;
      }

      console.log("Requesting geolocation...");
      setStatus(status, msg("fetching"));

      navigator.geolocation.getCurrentPosition(
        function (position) {
          console.log("Geolocation success:", position.coords);
          var lat = position.coords.latitude;
          var lon = position.coords.longitude;

          if (latInput) latInput.value = lat;
          if (lonInput) lonInput.value = lon;

          // Fetch weather
          console.log("Fetching weather for lat:", lat, "lon:", lon);
          fetch("/api/weather?lat=" + lat + "&lon=" + lon)
            .then(function (response) {
              console.log("Weather API response:", response.status);
              if (!response.ok) {
                throw new Error("Weather fetch failed: " + response.status);
              }
              return response.json();
            })
            .then(function (data) {
              console.log("Weather data:", data);
              if (tempInput) tempInput.value = data.temperature;
              if (humidityInput) humidityInput.value = data.humidity;
              setStatus(status, msg("weatherUpdated"));
            })
            .catch(function (error) {
              console.error("Weather fetch error:", error);
              setStatus(status, msg("weatherFailed"));
            });

          // Fetch soil
          console.log("Fetching soil for lat:", lat, "lon:", lon);
          fetch("/api/soil?lat=" + lat + "&lon=" + lon)
            .then(function (response) {
              console.log("Soil API response:", response.status);
              if (!response.ok) {
                throw new Error("Soil fetch failed: " + response.status);
              }
              return response.json();
            })
            .then(function (data) {
              console.log("Soil data:", data);
              if (nitrogenInput && data.nitrogen !== null) nitrogenInput.value = data.nitrogen;
              if (phosphorousInput && data.phosphorous !== null) phosphorousInput.value = data.phosphorous;
              if (pottasiumInput && data.pottasium !== null) pottasiumInput.value = data.pottasium;
              if (phInput && data.ph !== null) phInput.value = data.ph;
              if (soilTypeSelect && data.soil_type) soilTypeSelect.value = data.soil_type;
              setStatus(status, msg("soilUpdated"));
            })
            .catch(function (error) {
              console.error("Soil fetch error:", error);
              setStatus(status, msg("soilFailed"));
            });

          // Fetch city
          console.log("Fetching city for lat:", lat, "lon:", lon);
          fetch("/api/reverse-geocode?lat=" + lat + "&lon=" + lon)
            .then(function (response) {
              console.log("Reverse geocode API response:", response.status);
              if (!response.ok) {
                throw new Error("Reverse geocode failed: " + response.status);
              }
              return response.json();
            })
            .then(function (data) {
              console.log("City data:", data);
              if (cityInput && data.city) {
                cityInput.value = data.city;
                setStatus(status, msg("cityUpdated"));
              }
            })
            .catch(function (error) {
              console.error("Reverse geocode error:", error);
            });
        },
        function (error) {
          console.error("Geolocation error:", error);
          let errorMessage = msg("locationDenied");
          if (error.code === error.PERMISSION_DENIED) {
            errorMessage = "Location permission denied. Please allow location access and try again.";
          } else if (error.code === error.POSITION_UNAVAILABLE) {
            errorMessage = "Location information is unavailable. Please check your GPS/settings.";
          } else if (error.code === error.TIMEOUT) {
            errorMessage = "Location request timed out. Please try again.";
          }
          setStatus(status, errorMessage);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000
        }
      );
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindWeather("crop");
    bindWeather("fertilizer");
  });
})();
