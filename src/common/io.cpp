#include "common/io.h"

#include "3rd_party/cnpy/cnpy.h"
#include "common/shape.h"
#include "common/types.h"

#include "common/io_item.h"
#include "common/binary.h"


namespace marian {

namespace io {

bool isNpz(const std::string& fileName) {
  return fileName.size() >= 4 && fileName.substr(fileName.length() - 4) == ".npz";
}

bool isBin(const std::string& fileName) {
  return fileName.size() >= 4 && fileName.substr(fileName.length() - 4) == ".bin";
}

void getYamlFromNpz(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = cnpy::npz_load(fileName, varName);
  if(item->size() > 0)
    yaml = YAML::Load(item->data());
}

void getYamlFromBin(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = binary::getItem(fileName, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const std::string& fileName) {
  if(io::isNpz(fileName)) {
    io::getYamlFromNpz(yaml, varName, fileName);
  }
  else if(io::isBin(fileName)) {
    io::getYamlFromBin(yaml, varName, fileName);
  }
  else {
    ABORT("Unknown model file format for file {}", fileName);
  }
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const void* ptr) {
  auto item = binary::getItem(ptr, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
}

void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items) {
  Item item;

  item.name = varName;

  // increase size by 1 to add \0
  item.shape = Shape({(int)meta.size() + 1});

  item.bytes.resize(item.shape[0]);
  std::copy(meta.begin(), meta.end() + item.shape[0], item.bytes.begin());

  item.type = Type::int8;

  items.push_back(item);
}

void loadItemsFromNpz(const std::string& fileName, std::vector<Item>& items) {
    auto numpy = cnpy::npz_load(fileName);
    for(auto it : numpy) {

      Shape shape;
      if(it.second->shape.size() == 1) {
        shape.resize(2);
        shape.set(0, 1);
        shape.set(1, it.second->shape[0]);
      } else {
        shape.resize(it.second->shape.size());
        for(size_t i = 0; i < it.second->shape.size(); ++i)
          shape.set(i, it.second->shape[i]);
      }

      Item item;
      item.name = it.first;
      item.shape = shape;
      item.bytes.swap(it.second->bytes);

      items.emplace_back(std::move(item));
    }
}

std::vector<Item> loadItems(const std::string& fileName) {
  std::vector<Item> items;
  if(isNpz(fileName)) {
    loadItemsFromNpz(fileName, items);
  }
  else if(isBin(fileName)) {
    binary::loadItems(fileName, items);
  }
  else {
    ABORT("Unknown model file format for file {}", fileName);
  }

  return items;
}

std::vector<Item> loadItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, false);
  return items;
}

std::vector<Item> mmapItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, true);
  return items;
}

// @TODO: make cnpy and our wrapper talk to each other in terms of types
// or implement our own saving routines for npz based on npy, probably better.
void saveItemsNpz(const std::string& fileName, const std::vector<Item>& items) {
  std::vector<cnpy::NpzItem> npzItems;
  for(auto& item : items) {
    std::vector<unsigned int> shape(item.shape.begin(), item.shape.end());
    if(item.type == Type::float32)
      npzItems.push_back(cnpy::NpzItem(item.name,
                                       item.bytes,
                                       shape,
                                       cnpy::map_type(typeid(float)),
                                       sizeOf(Type::float32)));
    else if(item.type == Type::int8) {
      npzItems.push_back(cnpy::NpzItem(item.name,
                                       item.bytes,
                                       shape,
                                       cnpy::map_type(typeid(char)),
                                       sizeOf(Type::int8)));
    }
    else {
      ABORT("Type currently not supported");
    }
  }
  cnpy::npz_save(fileName, npzItems);
}

void saveItems(const std::string& fileName, const std::vector<Item>& items) {
  if(isNpz(fileName)) {
    saveItemsNpz(fileName, items);
  }
  else if(isBin(fileName)) {
    binary::saveItems(fileName, items);
  }
  else {
    ABORT("Unknown file format for file {}", fileName);
  }
}

}
}
