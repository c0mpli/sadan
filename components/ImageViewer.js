import { StyleSheet, Image } from "react-native";
import { useRoute } from "@react-navigation/native";

export default function ImageViewer({ placeholderImageSource, selectedImage }) {
  const imageSource = selectedImage
    ? { uri: selectedImage }
    : placeholderImageSource;
  const route = useRoute();

  return (
    <Image
      source={imageSource}
      style={route.name === "Home" ? styles.homeImage : styles.image}
    />
  );
}

const styles = StyleSheet.create({
  homeImage: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
  image: {
    width: 320,
    height: 180,
    borderRadius: 18,
  },
});
