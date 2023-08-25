import React, { useState } from "react";
import { Image, StyleSheet, Text, View } from "react-native";
import ImageViewer from "../components/ImageViewer";

export default function Results({ route }) {
  const { beforeImage, afterImage } = route.params;
  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <ImageViewer
          placeholderImageSource={beforeImage}
          selectedImage={beforeImage}
        />
      </View>
      <View style={styles.imageContainer}>
        <ImageViewer
          placeholderImageSource={afterImage}
          selectedImage={afterImage}
        />
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
