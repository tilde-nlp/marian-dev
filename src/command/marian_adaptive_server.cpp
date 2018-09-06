#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "data/adaptation_set.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

AdaptationSet parse(std::string message){
  //Parse the translation request (message) in JSON.
  YAML::Node message_node = YAML::Load(input);
  if (!message_node["text"]) {
    //TODO: Should we handle malformed data?
  }
  std::string text = message_node["text"].as<std::string>();
  std::vector<std::string> context_sources;
  std::vector<std::string> context_targets;
  //Check if context exists in the JSON and read it.
  if (message_node["context"]) {
    YAML::Node contexts_node = message_node["context"];
    for (std::size_t i=0; i < contexts_node.size(); i++) {
      YAML::Node context_node = contexts_node[i];
      if (context_node["source"] && context_node["target"]) {
        context_sources.push_back(context_node["source"].as<std::string>());
        context_targets.push_back(context_node["target"].as<std::string>());
      }
    }
  }
  return AdaptationSet(text, context_sources, context_targets);
  // TODO: member initialisation is probably wrong here
}

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WsServer;

int main(int argc, char **argv) {
  using namespace marian;

  // initialize translation model task
  auto options = New<Config>(argc, argv, ConfigMode::translating, true);
  auto task = New<TranslateService<BeamSearch>>(options);

  // create web service server
  WsServer server;
  server.config.port = options->get<size_t>("port");
  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&task](Ptr<WsServer::Connection> connection,
                                 Ptr<WsServer::Message> message) {
    auto message_str = message->string();

    auto message_short = message_str;
    boost::algorithm::trim_right(message_short);
    LOG(error, "Message received: {}", message_short);

    auto send_stream = std::make_shared<WsServer::SendStream>();
    boost::timer::cpu_timer timer;
    for(auto &transl : task->run({parse(message_str)})) {
      LOG(info, "Best translation: {}", transl);
      *send_stream << transl << std::endl;
    }
    LOG(info, "Translation took: {}", timer.format(5, "%ws"));

    connection->send(send_stream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        auto ec_str = std::to_string(ec.value());
        LOG(error, "Error sending message: ({}) {}", ec_str, ec.message());
      }
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WsServer::Connection> connection,
                          const SimpleWeb::error_code &ec) {
    auto ec_str = std::to_string(ec.value());
    LOG(error, "Connection error: ({}) {}", ec_str, ec.message());
  };

  // start server
  std::thread server_thread([&server]() {
    LOG(info, "Server is listening on port {}", server.config.port);
    server.start();
  });

  server_thread.join();

  return 0;
}
