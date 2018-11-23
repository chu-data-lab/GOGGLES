FROM rabbitmq:latest

EXPOSE 5672
EXPOSE 15672

COPY enabled_plugins /etc/rabbitmq/
COPY rabbitmq.config /etc/rabbitmq/

COPY init.sh /
COPY config_rabbit.sh /

RUN chmod 777 /init.sh /config_rabbit.sh

CMD ["/init.sh"]
