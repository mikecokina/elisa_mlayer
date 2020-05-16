import json

from elisa_mlayer import sio
from elisa_mlayer import config


extd_mysql_io = sio.get_mysqlio(config.DB_CONF, table_name="synthetic_lc", io_cls=sio.SyntheticFlatMySqlIO)


def kick_by_overcontact_teff(primary_t_eff, secondary_t_eff, morphology):
    if morphology in ["over-contact"]:
        if abs(primary_t_eff - secondary_t_eff) > 500:
            return True
    return False


if __name__ == "__main__":

    extd_mysql_io._preinit_method()
    _session = extd_mysql_io._get_session()

    loop_index, batch_size = 0, 10000
    while True:
        print(f"loop index: {loop_index}")

        result = _session.query(extd_mysql_io._model_instance) \
            .filter(extd_mysql_io._model_declarative_meta.morphology == "over-contact") \
            .with_entities(extd_mysql_io._model_declarative_meta.params, extd_mysql_io._model_declarative_meta.id) \
            .offset(loop_index * batch_size) \
            .limit(batch_size) \
            .all()

        loop_index += 1
        if len(result) == 0:
            break

        delete = list()
        for _row in result:
            row = json.loads(_row[0])
            idx = int(_row[1])
            kick = kick_by_overcontact_teff(row["primary"]["t_eff"], row["secondary"]["t_eff"], "over-contact")

            if kick:
                delete.append(idx)
                # _session.query(extd_mysql_io._model_declarative_meta).filter(extd_mysql_io._model_declarative_meta.id)\
                #     .delete(synchronize_session='fetch')

        if len(delete) > 0:
            _session.query(extd_mysql_io._model_declarative_meta)\
                .filter(extd_mysql_io._model_declarative_meta.id.in_(delete))\
                .delete(synchronize_session='fetch')
            _session.commit()
