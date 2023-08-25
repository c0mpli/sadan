import React from "react";
const PlaceholderImage = require("../assets/background-image.png");
import { useState } from "react";
import { StyleSheet, View } from "react-native";
import ImageViewer from "../components/ImageViewer";
import Button from "../components/Button";
import axios from "axios";
import * as ImagePicker from "expo-image-picker";
import { useNavigation } from "@react-navigation/native";
import LottieView from "lottie-react-native";

const BACKEND_URL = "http://192.168.1.83:5001";

export default function Home() {
  const navigation = useNavigation();
  const [selectedImage, setSelectedImage] = useState(null);
  const [afterImageLoaded, setAfterImageLoaded] = useState(true);
  const pickImageAsync = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    } else {
      alert("You did not select any image.");
    }
  };

  const submitImageAsync = async () => {
    setAfterImageLoaded(false);
    const formData = new FormData();
    formData.append("image", {
      uri: selectedImage,
      name: "image.jpg",
      type: "image/jpeg",
    });
    let response;
    try {
      response = await axios.post(`http://127.0.0.1:5001/recommend`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
    } catch (error) {
      alert("Something went wrong. Please try again.");
      console.log(error);
    } finally {
      setAfterImageLoaded(true);
      navigation.navigate("Results", {
        beforeImage: selectedImage,
        afterImage: response ? response?.data.output : selectedImage,
        afterImageLoaded: afterImageLoaded,
      });
    }
  };
  return !afterImageLoaded ? (
    <View style={styles.container}>
      <LottieView
        source={require("../assets/loading.json")}
        autoPlay
        loop
        style={{ width: 100, height: 100, marginVertical: 150 }}
      />
    </View>
  ) : (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <ImageViewer
          placeholderImageSource={PlaceholderImage}
          selectedImage={selectedImage}
        />
      </View>
      <View style={styles.footerContainer}>
        <Button
          theme="primary"
          label="Choose a photo"
          onPress={pickImageAsync}
        />
        <Button label="Upload photo" submitImageAsync={submitImageAsync} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
    alignItems: "center",
  },
  imageContainer: {
    flex: 1,
    paddingTop: 58,
  },
  image: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
});
